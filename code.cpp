/* 
 * Inference for Llama-2 Transformer model in pure C, int8 quantized forward pass.
 * 
 * 이 코드는 Llama-2 트랜스포머 모델의 int8 양자화 추론을 위한 순수 C/C++ 코드입니다.
 * 각 부분에 대해 주석을 추가하여 코드의 역할과 흐름을 설명합니다.
 */

#include <stdio.h>      // 표준 입출력 함수 (printf, fprintf 등)
#include <stdlib.h>     // 메모리 할당, exit, atoi 등 표준 라이브러리 함수
#include <ctype.h>      // 문자 판별 함수 (isprint, isspace 등)
#include <stdint.h>     // 고정 크기 정수 타입 (int8_t, uint32_t 등)
#include <time.h>       // 시간 측정 관련 함수 (clock_gettime 등)
#include <math.h>       // 수학 함수 (expf 등)
#include <string>       // C++ std::string 클래스
#include <iostream>     // C++ 표준 입출력 스트림
#include <cstring>      // 문자열 및 메모리 관련 함수 (memcpy, strlen 등)
#include <fcntl.h>      // 파일 제어 상수 및 함수 (open 등)
#include "typedefs.h"   // 사용자 정의 타입 선언
#include "forward.h"    // 추론(forward pass) 관련 함수 및 정의
#include "config.h"     // 모델 구성(Config) 관련 구조체 및 정의

#include <xrt/xrt_bo.h>        // XRT buffer object 관련 함수 및 타입
#include <xrt/xrt_device.h>    // XRT device 인터페이스
#include <xrt/xrt_kernel.h>    // XRT 커널 인터페이스
#if defined _WIN32
#include "win.h"               // 윈도우 관련 함수 및 헤더 (Windows 전용)
#else
#include <unistd.h>            // POSIX 시스템 함수
#include <sys/mman.h>          // 메모리 매핑(mmap) 관련 함수
#endif
// ----------------------------------------------------------------------------
// Globals
//
// (아래 주석 처리된 malloc_run_state 함수는 런타임 상태(RunState) 메모리 할당 예시)
// void malloc_run_state(RunState *s, Config *p)
// {
//   // 메모리 할당 시 calloc 사용 (valgrind 체크를 위해)
//   int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
//   s->x = (float *)calloc(p->dim, sizeof(float));
//   s->xb = (float *)calloc(p->dim, sizeof(float));
//   s->xb2 = (float *)calloc(p->dim, sizeof(float));
//   s->hb = (float *)calloc(p->hidden_dim, sizeof(float));
//   s->hb2 = (float *)calloc(p->hidden_dim, sizeof(float));
//   s->xq = (QuantizedTensor){.q = (int8_t *)calloc(p->dim, sizeof(int8_t)), .s = (float *)calloc(p->dim, sizeof(float))};
//   s->hq = (QuantizedTensor){.q = (int8_t *)calloc(p->hidden_dim, sizeof(int8_t)), .s = (float *)calloc(p->hidden_dim, sizeof(float))};
//   s->q = (float *)calloc(p->dim, sizeof(float));
//   s->k = (float *)calloc(kv_dim, sizeof(float));
//   s->v = (float *)calloc(kv_dim, sizeof(float));
//   s->att = (float *)calloc(p->n_heads * p->seq_len, sizeof(float));
//   s->logits = (float *)calloc(p->vocab_size, sizeof(float));
//   s->key_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
//   s->value_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
//   // 모든 할당이 성공했는지 확인
//   if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v || !s->att || !s->logits || !s->key_cache || !s->value_cache)
//   {
//     fprintf(stderr, "malloc failed!\n");
//     exit(EXIT_FAILURE);
//   }
// }

///////////////////////////////////////////////////////////////////////////////
// Softmax 계산 함수
//
// 입력 배열 x에 대해 수치 안정성을 고려하여 softmax를 계산합니다.
void softmax(float *x, int size)
{
  // 1. 수치 안정성을 위해 배열 내 최대값을 찾음
  float max_val = x[0];
  for (int i = 1; i < size; i++)
  {
    if (x[i] > max_val)
    {
      max_val = x[i];
    }
  }
  // 2. 각 값에 대해 exp(x - max_val)을 계산하고 전체 합(sum)을 구함
  float sum = 0.0f;
  for (int i = 0; i < size; i++)
  {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // 3. 각 값을 전체 합으로 나누어 확률 분포를 만듦
  for (int i = 0; i < size; i++)
  {
    x[i] /= sum;
  }
}

///////////////////////////////////////////////////////////////////////////////
// 템플릿 함수: 양자화 텐서를 초기화
//
// ptr가 가리키는 메모리에서 n개의 양자화 텐서를 size_each 요소씩 초기화합니다.
// 각 텐서는 int8 값(q)과 scale factor(s)를 포함합니다.
template <int SIZE>
void init_quantized_tensors(void **ptr, QuantizedTensor<SIZE> *tensor, int n, int size_each)
{
  void *p = *ptr;
  for (int i = 0; i < n; i++)
  {
    // int8 타입의 양자화 값 복사
    std::memcpy(tensor[i].q, p, size_each * sizeof(int8_t));
    p = (int8_t *)p + size_each;
    // scale factor (float) 복사
    std::memcpy(tensor[i].s, p, (size_each / GS) * sizeof(float));
    p = (float *)p + size_each / GS;
  }
  *ptr = p; // 포인터를 현재 위치로 업데이트
}

///////////////////////////////////////////////////////////////////////////////
// 템플릿 함수: 메모리 매핑을 통해 모델 가중치 로드
//
// 체크포인트 파일로부터 읽은 데이터를 FP32 가중치와 양자화된 텐서로 초기화합니다.
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void memory_map_weights(TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *w, void *ptr, uint8_t shared_classifier)
{
  int head_size = dim / n_heads;
  // 1. 먼저 FP32 가중치 (rmsnorm 관련)를 읽어옴
  float *fptr = (float *)ptr; // 포인터를 float*로 캐스팅
  std::memcpy(w->rms_att_weight, fptr, n_layers * dim * sizeof(float));
  fptr += n_layers * dim;
  std::memcpy(w->rms_ffn_weight, fptr, n_layers * dim * sizeof(float));
  fptr += n_layers * dim;
  std::memcpy(w->rms_final_weight, fptr, dim * sizeof(float));
  fptr += dim;

  // 2. 양자화된 가중치 데이터를 읽기 위해 포인터를 다시 void*로 설정
  ptr = (void *)fptr;
  // 토큰 임베딩 관련 양자화 텐서 초기화 후 dequantize 수행
  init_quantized_tensors(&ptr, w->q_tokens, 1, vocab_size * dim);
  dequantize<vocab_size * dim>(w->q_tokens, w->token_embedding_table, GS);

  // 각 레이어에 대해 쿼리, 키, 값, 출력 가중치 초기화
  init_quantized_tensors(&ptr, w->wq, n_layers, dim * (n_heads * head_size));
  init_quantized_tensors(&ptr, w->wk, n_layers, dim * (n_kv_heads * head_size));
  init_quantized_tensors(&ptr, w->wv, n_layers, dim * (n_kv_heads * head_size));
  init_quantized_tensors(&ptr, w->wo, n_layers, (n_heads * head_size) * dim);

  // 피드포워드 네트워크 가중치 초기화
  init_quantized_tensors(&ptr, w->w1, n_layers, dim * hidden_dim);
  init_quantized_tensors(&ptr, w->w2, n_layers, hidden_dim * dim);
  init_quantized_tensors(&ptr, w->w3, n_layers, dim * hidden_dim);

  // 분류기(classifier) 가중치 처리: 공유된 경우와 별도 초기화하는 경우 분기
  if (shared_classifier)
  {
    std::memcpy(w->wcls, w->q_tokens, sizeof(QuantizedTensor<vocab_size * dim>));
  }
  else
  {
    init_quantized_tensors(&ptr, w->wcls, 1, dim * vocab_size);
  }
}

///////////////////////////////////////////////////////////////////////////////
// 템플릿 함수: 체크포인트 파일을 읽어 모델 구성(Config) 및 가중치 로드
//
// 파일을 열어 magic number, 버전, Config, 플래그, 그룹 크기 등을 확인한 후,
// mmap을 사용해 전체 파일을 메모리에 매핑하고, 메모리 매핑된 데이터를 통해 가중치를 로드합니다.
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void read_checkpoint(std::string checkpoint, Config *config, TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *weights)
{
  // 체크포인트 파일 열기
  FILE *file = fopen(checkpoint.c_str(), "rb");
  if (!file)
  {
    fprintf(stderr, "Couldn't open file %s\n", checkpoint.c_str());
    exit(EXIT_FAILURE);
  }
  // 매직 넘버 확인 ("ak42"에 해당하는 0x616b3432)
  uint32_t magic_number;
  if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1)
  {
    exit(EXIT_FAILURE);
  }
  if (magic_number != 0x616b3432)
  {
    fprintf(stderr, "Bad magic number\n");
    exit(EXIT_FAILURE);
  }
  // 버전 번호 읽기 (버전 2가 필요함)
  int version;
  if (fread(&version, sizeof(int), 1, file) != 1)
  {
    exit(EXIT_FAILURE);
  }
  if (version != 2)
  {
    fprintf(stderr, "Bad version %d, need version 2\n", version);
    exit(EXIT_FAILURE);
  }
  int header_size = 256; // 버전 2 헤더의 크기는 256바이트
  // Config 구조체 읽기 (구성 정보)
  if (fread(config, sizeof(Config) - sizeof(int), 1, file) != 1)
  {
    exit(EXIT_FAILURE);
  }
  // 플래그 읽기: classifier 공유 여부
  uint8_t shared_classifier;
  if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1)
  {
    exit(EXIT_FAILURE);
  }
  // 양자화 그룹 크기 읽기
  int group_size;
  if (fread(&group_size, sizeof(int), 1, file) != 1)
  {
    exit(EXIT_FAILURE);
  }
  config->GS = GS;
  // 파일 전체 크기 확인
  fseek(file, 0, SEEK_END);
  auto file_size = ftell(file);
  fclose(file);
  // mmap을 사용하여 체크포인트 파일 전체를 메모리에 매핑
  auto fd = open(checkpoint.c_str(), O_RDONLY);
  if (fd == -1)
  {
    fprintf(stderr, "open failed!\n");
    exit(EXIT_FAILURE);
  }
  auto data = (float *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (data == MAP_FAILED)
  {
    fprintf(stderr, "mmap failed!\n");
    exit(EXIT_FAILURE);
  }
  // 헤더 부분(256바이트)을 건너뛰고 가중치 데이터에 대한 포인터 설정
  void *weights_ptr = ((char *)data) + header_size;
  // 메모리 매핑된 데이터를 통해 모델 가중치 초기화
  memory_map_weights(weights, weights_ptr, shared_classifier);
  close(fd);
  if (data != MAP_FAILED)
  {
    munmap(data, file_size);
  }
}

///////////////////////////////////////////////////////////////////////////////
// 템플릿 함수: Transformer 객체 생성
//
// 체크포인트 파일 경로를 받아 모델 구성(Config)과 가중치를 로드하여 Transformer 객체를 구성합니다.
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void build_transformer(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *t, std::string checkpoint_path)
{
  // read_checkpoint 함수를 호출하여 모델 구성과 가중치를 로드함
  read_checkpoint(checkpoint_path, &t->config, &t->weights);
}

///////////////////////////////////////////////////////////////////////////////
// BPE (Byte Pair Encoding) 토크나이저 관련 구조체 및 함수

// TokenIndex 구조체: 토큰 문자열과 해당 ID를 저장 (정렬 및 검색 용도)
typedef struct
{
  char *str;
  int id;
} TokenIndex;

// Tokenizer 구조체: 전체 vocabulary, 각 토큰의 점수, 정렬된 vocab, 최대 토큰 길이, 단일 바이트 문자열 저장
typedef struct
{
  char **vocab;
  float *vocab_scores;
  TokenIndex *sorted_vocab;
  int vocab_size;
  unsigned int max_token_length;
  unsigned char byte_pieces[512]; // 0~255까지의 단일 바이트 문자열 저장
} Tokenizer;

///////////////////////////////////////////////////////////////////////////////
// 토큰 비교 함수: qsort, bsearch 등에서 사용 (문자열 비교)
int compare_tokens(const void *a, const void *b)
{
  return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

///////////////////////////////////////////////////////////////////////////////
// 토크나이저 빌드 함수: tokenizer 파일을 읽어 Tokenizer 구조체 초기화
void build_tokenizer(Tokenizer *t, std::string tokenizer_path, int vocab_size)
{
  // vocab 크기 설정
  t->vocab_size = vocab_size;
  // vocabulary, 점수, 정렬된 vocab에 필요한 메모리 할당
  t->vocab = (char **)malloc(vocab_size * sizeof(char *));
  t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
  t->sorted_vocab = NULL; // 정렬은 필요 시 지연 초기화
  // 0~255까지의 단일 바이트 토큰 초기화
  for (int i = 0; i < 256; i++)
  {
    t->byte_pieces[i * 2] = (unsigned char)i;
    t->byte_pieces[i * 2 + 1] = '\0';
  }
  // tokenizer 파일 열기
  FILE *file = fopen(tokenizer_path.c_str(), "rb");
  if (!file)
  {
    fprintf(stderr, "couldn't load %s\n", tokenizer_path.c_str());
    exit(EXIT_FAILURE);
  }
  // 최대 토큰 길이 읽기
  if (fread(&t->max_token_length, sizeof(int), 1, file) != 1)
  {
    fprintf(stderr, "failed read\n");
    exit(EXIT_FAILURE);
  }
  int len;
  // 각 토큰의 점수와 문자열을 읽어들임
  for (int i = 0; i < vocab_size; i++)
  {
    if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1)
    {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    if (fread(&len, sizeof(int), 1, file) != 1)
    {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i] = (char *)malloc(len + 1);
    if (fread(t->vocab[i], len, 1, file) != 1)
    {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i][len] = '\0'; // 널 종료 문자 추가
  }
  fclose(file);
}

///////////////////////////////////////////////////////////////////////////////
// 토크나이저 메모리 해제 함수
void free_tokenizer(Tokenizer *t)
{
  for (int i = 0; i < t->vocab_size; i++)
  {
    free(t->vocab[i]);
  }
  free(t->vocab);
  free(t->vocab_scores);
  free(t->sorted_vocab);
}

///////////////////////////////////////////////////////////////////////////////
// 디코드 함수: 토큰을 문자열 조각으로 변환
//
// 이전 토큰(prev_token)과 현재 토큰(token)을 기반으로 적절한 문자열 조각 반환
char *decode(Tokenizer *t, int prev_token, int token)
{
  char *piece = t->vocab[token];
  // BOS 토큰(1) 뒤의 토큰에서 선행 공백 제거 (SentencePiece 특성)
  if (prev_token == 1 && piece[0] == ' ')
  {
    piece++;
  }
  // 만약 토큰이 <0xXX> 형식이면, 실제 바이트 값으로 변환하여 반환
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1)
  {
    piece = (char *)t->byte_pieces + byte_val * 2;
  }
  return piece;
}

///////////////////////////////////////////////////////////////////////////////
// 안전 출력 함수: 출력 시 비프린터블 문자나 제어문자 필터링
void safe_printf(char *piece)
{
  if (piece == NULL)
  {
    return;
  }
  if (piece[0] == '\0')
  {
    return;
  }
  if (piece[1] == '\0')
  {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val)))
    {
      return; // 출력하지 않을 문자면 건너뜀
    }
  }
  printf("%s", piece);
}

///////////////////////////////////////////////////////////////////////////////
// 정렬된 vocab에서 문자열 검색 (이진 검색)
// 일치하는 토큰의 인덱스를 반환하며, 없으면 -1 반환
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size)
{
  TokenIndex tok = {.str = str}; // 검색 키 생성
  TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}

///////////////////////////////////////////////////////////////////////////////
// 인코딩 함수: 입력 문자열을 토큰 배열로 변환 (BPE 기반)
// bos, eos 플래그에 따라 BOS/EOS 토큰 추가 후 UTF-8 및 병합 처리를 수행
void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens)
{
  if (text == NULL)
  {
    fprintf(stderr, "cannot encode NULL text\n");
    exit(EXIT_FAILURE);
  }

  // 정렬된 vocab이 아직 초기화되지 않았다면, 초기화 후 정렬
  if (t->sorted_vocab == NULL)
  {
    t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < t->vocab_size; i++)
    {
      t->sorted_vocab[i].str = t->vocab[i];
      t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }

  // 병합 후보를 위한 임시 버퍼 할당 (최대 토큰 길이 고려)
  char *str_buffer = (char *)malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
  size_t str_len = 0;

  // 초기 토큰 개수를 0으로 설정
  *n_tokens = 0;

  // bos 플래그가 true면 BOS 토큰(1) 추가
  if (bos)
    tokens[(*n_tokens)++] = 1;

  // 입력 텍스트가 비어있지 않다면 dummy prefix(공백) 토큰 추가
  if (text[0] != '\0')
  {
    int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
    tokens[(*n_tokens)++] = dummy_prefix;
  }

  // UTF-8 바이트 스트림 처리: 각 코드포인트별로 분리하여 토큰화 수행
  for (char *c = text; *c != '\0'; c++)
  {
    // ASCII 또는 UTF-8 리딩의 첫 바이트인 경우 버퍼 리셋
    if ((*c & 0xC0) != 0x80)
    {
      str_len = 0;
    }
    // 현재 바이트를 버퍼에 추가
    str_buffer[str_len++] = *c;
    str_buffer[str_len] = '\0';

    // 다음 바이트가 연속 바이트인지 확인 (최대 4바이트까지)
    if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4)
    {
      continue;
    }

    // 완성된 코드포인트에 대해 vocabulary에서 토큰 검색
    int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

    if (id != -1)
    {
      // 토큰이 존재하면 토큰 배열에 추가
      tokens[(*n_tokens)++] = id;
    }
    else
    {
      // 존재하지 않으면 fallback으로 각 바이트를 개별 토큰으로 추가 (index에 3을 더함)
      for (int i = 0; i < str_len; i++)
      {
        tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
      }
    }
    str_len = 0; // 다음 코드포인트 처리를 위해 버퍼 리셋
  }

  // BPE merge: 인접 토큰 병합 규칙을 반복 적용하여 최적의 토큰 시퀀스로 구성
  while (1)
  {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;

    for (int i = 0; i < (*n_tokens - 1); i++)
    {
      // 두 인접 토큰을 병합한 문자열 생성
      sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
      int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
      if (id != -1 && t->vocab_scores[id] > best_score)
      {
        best_score = t->vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }

    if (best_idx == -1)
    {
      break; // 더 이상 병합 가능한 토큰이 없으면 종료
    }

    // 최적의 병합 후보를 선택하여 해당 위치의 토큰을 교체
    tokens[best_idx] = best_id;
    // 병합 후 배열에서 토큰 삭제 (shift)
    for (int i = best_idx + 1; i < (*n_tokens - 1); i++)
    {
      tokens[i] = tokens[i + 1];
    }
    (*n_tokens)--; // 토큰 수 감소
  }

  // eos 플래그가 true면 EOS 토큰(2) 추가
  if (eos)
    tokens[(*n_tokens)++] = 2;

  free(str_buffer);
}

///////////////////////////////////////////////////////////////////////////////
// 샘플러 및 토큰 선택 관련 코드

// ProbIndex 구조체: 토큰 인덱스와 해당 확률 저장 (top-p 샘플링에 사용)
typedef struct
{
  float prob;
  int index;
} ProbIndex;

// Sampler 구조체: vocab 크기, 샘플링 관련 파라미터(temperature, topp), 난수 생성 상태 등을 저장
typedef struct
{
  int vocab_size;
  ProbIndex *probindex; // top-p 샘플링 시 후보 저장용 버퍼
  float temperature;    // 온도 (sampling randomness)
  float topp;           // top-p (nucleus) 샘플링의 p 값
  unsigned long long rng_state; // xorshift 난수 생성기 상태
} Sampler;

///////////////////////////////////////////////////////////////////////////////
// 그리디 샘플링 함수: 가장 높은 확률을 가진 토큰 선택
int sample_argmax(float *probabilities, int n)
{
  int max_i = 0;
  float max_p = probabilities[0];
  for (int i = 1; i < n; i++)
  {
    if (probabilities[i] > max_p)
    {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

///////////////////////////////////////////////////////////////////////////////
// 확률적 샘플링 함수: 누적 분포(CDF)를 이용해 무작위로 토큰 선택
int sample_mult(float *probabilities, int n, float coin)
{
  float cdf = 0.0f;
  for (int i = 0; i < n; i++)
  {
    cdf += probabilities[i];
    if (coin < cdf)
    {
      return i;
    }
  }
  return n - 1; // rounding error 방지
}

///////////////////////////////////////////////////////////////////////////////
// qsort에서 사용되는 비교 함수: ProbIndex 내 확률 내림차순 정렬
int compare(const void *a, const void *b)
{
  ProbIndex *a_ = (ProbIndex *)a;
  ProbIndex *b_ = (ProbIndex *)b;
  if (a_->prob > b_->prob)
    return -1;
  if (a_->prob < b_->prob)
    return 1;
  return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Top-p (nucleus) 샘플링 함수
//
// 후보 토큰들 중 누적 확률이 topp 값을 넘을 때까지의 최소 집합에서 샘플링
int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin)
{
  int n0 = 0;
  // cutoff 기준보다 작은 확률을 미리 제외
  const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++)
  {
    if (probabilities[i] >= cutoff)
    {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }
  // 후보들을 내림차순으로 정렬
  qsort(probindex, n0, sizeof(ProbIndex), compare);

  // 누적 확률이 topp 값을 초과할 때까지 후보 집합 축소
  float cumulative_prob = 0.0f;
  int last_idx = n0 - 1; // rounding error를 위해 전체 후보 사용
  for (int i = 0; i < n0; i++)
  {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp)
    {
      last_idx = i;
      break;
    }
  }

  // 축소된 후보 집합 내에서 무작위 샘플링
  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++)
  {
    cdf += probindex[i].prob;
    if (r < cdf)
    {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index;
}

///////////////////////////////////////////////////////////////////////////////
// 샘플러 초기화 함수: Sampler 구조체 설정 및 내부 버퍼 할당
void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed)
{
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  // top-p 샘플링에 사용할 후보 배열 할당
  sampler->probindex = (ProbIndex *)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

///////////////////////////////////////////////////////////////////////////////
// 샘플러 메모리 해제 함수
void free_sampler(Sampler *sampler)
{
  free(sampler->probindex);
}

///////////////////////////////////////////////////////////////////////////////
// xorshift 알고리즘을 이용한 난수 생성: 32비트 무작위 정수 생성
unsigned int random_u32(unsigned long long *state)
{
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

///////////////////////////////////////////////////////////////////////////////
// [0, 1) 범위의 float 난수 생성 (xorshift 기반)
float random_f32(unsigned long long *state)
{
  return (random_u32(state) >> 8) / 16777216.0f;
}

///////////////////////////////////////////////////////////////////////////////
// 샘플 함수: 로짓을 입력받아 다음 토큰을 선택 (온도 및 top-p 적용)
//
// 온도가 0이면 그리디 샘플링, 그 외에는 확률적/Top-p 샘플링을 사용합니다.
int sample(Sampler *sampler, float *logits)
{
  int next;
  if (sampler->temperature == 0.0f)
  {
    // 온도 0이면 가장 높은 확률의 토큰 선택 (그리디)
    next = sample_argmax(logits, sampler->vocab_size);
  }
  else
  {
    // 온도 적용: 각 로짓 값을 스케일 조정
    for (int q = 0; q < sampler->vocab_size; q++)
    {
      logits[q] /= sampler->temperature;
    }
    // softmax 적용하여 확률 분포 생성
    softmax(logits, sampler->vocab_size);
    // 난수 값 생성 (0 <= coin < 1)
    float coin = random_f32(&sampler->rng_state);
    // topp 값에 따라 샘플링 방식 결정
    if (sampler->topp <= 0 || sampler->topp >= 1)
    {
      // topp가 비활성화된 경우 일반 확률적 샘플링
      next = sample_mult(logits, sampler->vocab_size, coin);
    }
    else
    {
      // topp (nucleus) 샘플링 적용
      next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
    }
  }
  return next;
}

///////////////////////////////////////////////////////////////////////////////
// 시간 측정 함수: 현재 시간을 밀리초 단위로 반환
long time_in_ms()
{
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

///////////////////////////////////////////////////////////////////////////////
// 생성(추론) 루프 함수
//
// Transformer 모델, Tokenizer, Sampler를 이용하여 주어진 프롬프트로부터 토큰을 생성하고 출력합니다.
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void generate(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps, std::string &kernelpath)
{
  char *empty_prompt = "";
  if (prompt == NULL)
  {
    prompt = empty_prompt;
  }

  // 1. 프롬프트 문자열을 토큰으로 인코딩 (BOS 토큰 포함)
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int)); // 추가 공간: BOS, EOS 등
  encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
  if (num_prompt_tokens < 1)
  {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // 2. XRT 디바이스에서 실행할 커널 로드 및 버퍼 할당
  std::cout << "Loading kernel..." << std::endl;
  auto device = xrt::device(0);
  auto uuid = device.load_xclbin(kernelpath);
  auto kernel = xrt::kernel(device, uuid, "forward");
  std::cout << "Out buffer size: " << vocab_size * sizeof(float) << std::endl;
  std::cout << "Transformer size: " << sizeof(*transformer) << std::endl;
  std::cout << "Allocating output buffer" << std::endl;
  auto out_buffer = xrt::bo(device, vocab_size * sizeof(float), kernel.group_id(5));

  // 키/값 캐시 버퍼 할당 (cache_dim 계산)
  int cache_dim = n_layers * seq_len * ((dim * n_kv_heads) / n_heads);
  std::cout << "Allocating buffers" << std::endl;
  auto transformer_buffer = xrt::bo(device, sizeof(*transformer), kernel.group_id(0));

  auto key_buffer = xrt::bo(device, cache_dim * sizeof(float), kernel.group_id(3));
  auto value_buffer = xrt::bo(device, cache_dim * sizeof(float), kernel.group_id(4));

  std::cout << "Copying data to buffer" << std::endl;
  // Transformer 모델 데이터를 디바이스 버퍼에 복사
  transformer_buffer.write(transformer, sizeof(*transformer), 0);
  transformer_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // 3. 첫 번째 커널 실행: 프롬프트의 첫 토큰으로 순전파 실행
  long start = 0;
  int next;
  int token = prompt_tokens[0]; // 프롬프트의 첫 토큰 사용
  int pos = 0;
  auto run = kernel(transformer_buffer, token, pos, key_buffer, value_buffer, out_buffer);
  run.wait();

  // 출력 버퍼에서 로짓 값을 읽어옴
  float *logits = (float *)malloc(vocab_size * sizeof(float));
  out_buffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  out_buffer.read(logits, vocab_size * sizeof(float), 0);

  // 프롬프트 내 토큰 순차 처리: 아직 프롬프트 토큰이 남아 있으면 강제 사용
  if (pos < num_prompt_tokens - 1)
  {
    next = prompt_tokens[pos + 1];
  }
  else
  {
    next = sample(sampler, logits);
  }
  pos++;

  // 첫 토큰 출력: 토큰을 디코딩하여 안전하게 출력
  char *piece = decode(tokenizer, token, next);
  safe_printf(piece);
  fflush(stdout);
  token = next;
  start = time_in_ms();

  // 4. 추론 루프: 지정된 step 수만큼 토큰 생성
  while (pos < steps)
  {
    run.set_arg(1, token);
    run.set_arg(2, pos);
    run.start();
    run.wait();

    out_buffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    out_buffer.read(logits, vocab_size * sizeof(float), 0);

    if (pos < num_prompt_tokens - 1)
    {
      next = prompt_tokens[pos + 1];
    }
    else
    {
      next = sample(sampler, logits);
    }
    pos++;

    // BOS 토큰(1)이 나오면 종료 (문장 구분 조건)
    if (next == 1)
    {
      break;
    }

    // 생성된 토큰을 디코딩하여 출력
    char *piece = decode(tokenizer, token, next);
    safe_printf(piece);
    fflush(stdout);
    token = next;
  }
  printf("\n");

  // 생성된 토큰 수와 시간 기반 토큰 생성 속도(tok/s) 출력
  if (pos > 1)
  {
    long end = time_in_ms();
    fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
  }

  free(prompt_tokens);
}

///////////////////////////////////////////////////////////////////////////////
// 표준 입력에서 한 줄을 읽어 지정된 버퍼에 저장 (개행문자 제거)
void read_stdin(const char *guide, char *buffer, size_t bufsize)
{
  printf("%s", guide);
  if (fgets(buffer, bufsize, stdin) != NULL)
  {
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n')
    {
      buffer[len - 1] = '\0';
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// CLI (명령행 인터페이스) 및 메인 함수
// TESTING 매크로가 정의되지 않은 경우에만 컴파일됨
#ifndef TESTING

// 사용법 에러 및 도움말 출력 함수
void error_usage()
{
  fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
  fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
  fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
  fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
  fprintf(stderr, "  -i <string> input prompt\n");
  fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
  fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  exit(EXIT_FAILURE);
}

///////////////////////////////////////////////////////////////////////////////
// 메인 함수: 인자 파싱 후 모델, 토크나이저, 샘플러 초기화 및 생성 루프 실행
int main(int argc, char *argv[])
{
  std::cout << "start" << std::endl;
  // 기본 인자 설정
  std::string checkpoint_path = "";       // 체크포인트 파일 경로
  std::string tokenizer_path = "tokenizer.bin";  // 토크나이저 파일 경로
  float temperature = 1.0f;                 // 온도 (0.0이면 그리디 샘플링)
  float topp = 0.9f;                        // top-p 샘플링의 p 값
  int steps = 256;                          // 생성할 최대 토큰 수
  char *prompt = NULL;                      // 입력 프롬프트 문자열
  unsigned long long rng_seed = 0;          // 난수 생성 시드
  const char *mode = "generate";            // 모드: generate 또는 chat (여기서는 generate)
  char *system_prompt = NULL;               // chat 모드에서 사용될 시스템 프롬프트 (옵션)
  std::string kernelpath = "";              // XRT 커널 바이너리 경로

  // 최소 2개 이상의 인자(체크포인트 파일 경로)가 필요함
  if (argc >= 2)
  {
    checkpoint_path = argv[1];
  }
  else
  {
    std::cout << "test1" << std::endl;
    error_usage();
  }
  // 인자 파싱: 각 옵션에 대해 두 개씩 (플래그와 값)
  for (int i = 2; i < argc; i += 2)
  {
    if (i + 1 >= argc)
    {
      error_usage();
    }
    if (argv[i][0] != '-')
    {
      error_usage();
    }
    if (strlen(argv[i]) != 2)
    {
      error_usage();
    }
    // 각 옵션에 따른 인자 처리
    if (argv[i][1] == 't')
    {
      temperature = atof(argv[i + 1]);
    }
    else if (argv[i][1] == 'p')
    {
      topp = atof(argv[i + 1]);
    }
    else if (argv[i][1] == 's')
    {
      rng_seed = atoi(argv[i + 1]);
    }
    else if (argv[i][1] == 'n')
    {
      steps = atoi(argv[i + 1]);
    }
    else if (argv[i][1] == 'i')
    {
      prompt = argv[i + 1];
    }
    else if (argv[i][1] == 'z')
    {
      tokenizer_path = argv[i + 1];
    }
    else if (argv[i][1] == 'm')
    {
      mode = argv[i + 1];
    }
    else if (argv[i][1] == 'y')
    {
      system_prompt = argv[i + 1];
    }
    else if (argv[i][1] == 'k')
    {
      kernelpath = argv[i + 1];
    }
    else
    {
      error_usage();
    }
  }

  // 인자 유효성 검사 및 기본값 적용
  if (rng_seed <= 0)
    rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0)
    temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp)
    topp = 0.9;
  if (steps < 0)
    steps = 0;

  // Transformer 모델 생성: 체크포인트 파일로부터 모델 구성 및 가중치 로드
  static Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> transformer;
  build_transformer(&transformer, checkpoint_path);
  if (steps == 0 || steps > transformer.config.seq_len)
    steps = transformer.config.seq_len;

  // 토크나이저 구성: tokenizer 파일로부터 vocab, 점수 등 초기화
  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

  // 샘플러 생성: 온도, topp, 난수 시드 설정
  Sampler sampler;
  build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

  // 모드에 따라 생성 또는 채팅 모드 실행 (여기서는 generate 모드만 처리)
  if (strcmp(mode, "generate") == 0)
  {
    generate(&transformer, &tokenizer, &sampler, prompt, steps, kernelpath);
  }
  else
  {
    fprintf(stderr, "unknown mode: %s\n", mode);
    error_usage();
  }

  // 사용한 메모리 해제
  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  return 0;
}
#endif
