/*
 * forward.h - Llama2 Transformer forward pass (HLS C/C++ for FPGA)
 *
 * 이 코드는 Llama2 같은 트랜스포머 모델의 한 단계를 FPGA에서 계산하기 위해
 * HLS(Hardware Language Synthesis) 방식으로 작성된 예시입니다.
 * 주석을 통해 각 함수가 어떤 역할을 하는지 설명합니다.
 */

#include "forward.h"  // Transformer, quantize, dim 등 정의
#include "config.h"   // config 구조체(하이퍼파라미터) 정의
#include <cstring>    // std::memcpy 등 사용

// TODO: include HLS math package
// (필요하다면 HLS 전용 math 헤더를 추가하여 sqrtf, expf, powf 등 최적화된 함수를 사용)

//---------------------------------------------------------------------------------------
// 1) RMSNorm 함수
//    - 입력 x[S]를 정규화하여 출력 o[S]에 저장
//    - weight[S]는 스케일링 파라미터(레이어 노름 파라미터)
//    - sum_of_squares 루프에서 벡터의 제곱합을 구하고, 그 루트를 사용해 정규화
//    - norm_and_scale 루프에서 weight를 곱해 최종 출력
//---------------------------------------------------------------------------------------
template <int S>
void rmsnorm(float o[S], float x[S], float weight[S])
{
  // 한 번에 복사할 바이트 크기
  constexpr auto array_size = S * sizeof(float);

  // ss는 x의 각 원소 제곱합 / S, 최종적으로 1 / sqrt(...)
  float ss = 0.0f;

  // 중간 버퍼
  float x_buff[S];
  float weight_buff[S];
  float out_buff[S];

  // HLS 지시어: 배열을 특정 factor로 파이프라인/언롤링/파티셔닝 최적화
#pragma HLS array_partition variable = x_buff type = cyclic factor = 128
#pragma HLS array_partition variable = weight_buff type = cyclic factor = 64
#pragma HLS array_partition variable = out_buff type = cyclic factor = 64

  // 입력 데이터를 x_buff, weight_buff로 복사
  std::memcpy(x_buff, x, array_size);
  std::memcpy(weight_buff, weight, array_size);

sum_of_squares:
  for (int j = 0; j < S; j++)
  {
#pragma HLS PIPELINE
#pragma HLS UNROLL factor = 128 skip_exit_check
    float x_j = x_buff[j];
    ss += x_j * x_j;  // 제곱합 누적
  }

  // 평균 내고, epsilon(1e-5) 더한 뒤, 역수의 sqrt()를 취함
  ss /= S;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);

norm_and_scale:
  for (int j = 0; j < S; j++)
  {
#pragma HLS PIPELINE
#pragma HLS UNROLL factor = 64
    float weight_j = weight_buff[j];
    float x_j = x_buff[j];
    // x_j를 정규화한 값에 weight_j를 곱해 최종 out
    out_buff[j] = weight_j * (ss * x_j);
  }

  // 최종 결과 out_buff -> o로 복사
  std::memcpy(o, out_buff, array_size);
}

//---------------------------------------------------------------------------------------
// 2) softmax 함수
//    - x 배열에서 최대값을 찾아 exp(x - max_val)을 계산
//    - 합(sum)을 구해 각 항을 sum으로 나눈다
//    - MAXSIZE는 최대 크기를 템플릿으로 설정하여 HLS 최적화에 활용
//---------------------------------------------------------------------------------------
template <int MAXSIZE>
void softmax(float *x, int size)
{
  // 중간 버퍼(각 원소의 exp 결과를 저장)
  float buffer[MAXSIZE];

  // 1) x 배열에서 최대값(max_val) 찾기
  float max_val = x[0];
max:
  for (int i = 1; i < size; i++)
  {
#pragma HLS loop_tripcount min = 0 max = 257 avg = 129
#pragma HLS PIPELINE
    float x_i = x[i];
    if (x_i > max_val)
    {
      max_val = x_i;
    }
  }

  // 2) exp & sum
exp:
  for (int i = 0; i < size; i++)
  {
#pragma HLS loop_tripcount min = 0 max = 257 avg = 129
#pragma HLS PIPELINE
#pragma HLS UNROLL factor = 16
    float x_i = expf(x[i] - max_val); // 안정성을 위해 max_val 빼기
    buffer[i] = x_i;
  }
  float sum = 0.0f;
sum:
  for (int i = 0; i < size; i++)
  {
#pragma HLS loop_tripcount min = 0 max = 257 avg = 129
    sum += buffer[i];
  }

  // 3) 각 항을 sum으로 나누어 softmax 확률 분포로 만든다
  const float inv_sum = 1.0f / sum;
norm:
  for (int i = 0; i < size; i++)
  {
#pragma HLS loop_tripcount min = 0 max = 257 avg = 129
#pragma HLS PIPELINE
#pragma HLS UNROLL factor = 16
    x[i] = buffer[i] * inv_sum;
  }
}

//---------------------------------------------------------------------------------------
// 3) matmul_old
//    - 이전에 작성된(또는 실험용) 행렬 곱 함수
//    - W(d,n) @ x(n,) -> xout(d,)
//    - wq, xq는 int8, ws, xs는 float(scale factor)
//    - x_buffer, w_buffer 등 중간 버퍼를 만들어 HLS에서 최적화
//---------------------------------------------------------------------------------------
template <int N, int D>
void matmul_old(float *xout, int8_t *xq, float *xs, int8_t *wq, float *ws)
{
  // W: (D, N), x: (N,)  => xout: (D,)

  // 중간 버퍼: 입력 벡터, 스케일, 가중치, 가중치 스케일
  static int8_t x_buffer[N];
  static float xs_buffer[N / GS];
  int8_t w_buffer[N * D];
  float ws_buffer[N * D / GS];

#pragma HLS ARRAY_PARTITION variable = x_buffer type = cyclic factor = 16
#pragma HLS ARRAY_PARTITION variable = xs_buffer type = cyclic factor = 4
#pragma HLS ARRAY_PARTITION variable = w_buffer type = cyclic factor = 128
#pragma HLS ARRAY_PARTITION variable = ws_buffer type = cyclic factor = 32

  // xq, xs, wq, ws 데이터를 버퍼에 복사
x_buff:
  for (int i = 0; i < N; i++)
  {
#pragma HLS UNROLL factor = 16
    x_buffer[i] = xq[i];
  }
xs_buff:
  for (int j = 0; j <= N - GS; j += GS)
  {
#pragma HLS UNROLL factor = 4
    xs_buffer[j / GS] = xs[j / GS];
  }

w_buff:
  for (int i = 0; i < N * D; i++)
  {
#pragma HLS UNROLL factor = 128
    w_buffer[i] = wq[i];
  }

ws_buff:
  for (int i = 0; i < N * D / GS; i++)
  {
#pragma HLS UNROLL factor = 32
    ws_buffer[i] = ws[i];
  }

  // 실제 matmul 계산
  for (int i = 0; i < D; i++)
  {
#pragma HLS PIPEPLINE
    float val = 0.0f;
    const int in = i * N;       // i번째 row의 시작 인덱스
    const int in_s = i * N / GS; // 스케일 배열에서의 오프셋

  matmul3:
    for (int j = 0; j <= N - GS; j += GS)
    {
#pragma HLS UNROLL
      int32_t ival = 0;
    matmul4:
      for (int k = 0; k < GS; k++)
      {
#pragma HLS UNROLL
        // x_buffer[j + k] * w_buffer[in + j + k]
        ival += ((int32_t)x_buffer[j + k]) * ((int32_t)w_buffer[in + j + k]);
      }
      // int32 누적값에 scale factor 적용
      val += ((float)ival) * ws_buffer[in_s + j / GS] * xs_buffer[j / GS];
    }
    xout[i] = val;
  }
}

//---------------------------------------------------------------------------------------
// 4) matmul
//    - 실제 사용되는 행렬 곱 함수(최적화 버전)
//    - W(d,n) @ x(n,) -> xout(d,)
//    - wq, xq는 int8, ws, xs는 float(scale factor)
//    - 내부에서 row 단위로 w_buffer, ws_buffer를 읽어 matmul 수행
//---------------------------------------------------------------------------------------
template <int N, int D>
void matmul(float *xout, int8_t *xq, float *xs, int8_t *wq, float *ws)
{
  // 중간 버퍼
  static int8_t x_buffer[N];
  static float xs_buffer[N / GS];

#pragma HLS ARRAY_PARTITION variable = x_buffer type = cyclic factor = 16
#pragma HLS ARRAY_PARTITION variable = xs_buffer type = cyclic factor = 4

  // xq, xs -> x_buffer, xs_buffer
x_buff:
  for (int i = 0; i < N; i++)
  {
#pragma HLS UNROLL factor = 16
    x_buffer[i] = xq[i];
  }
xs_buff:
  for (int j = 0; j <= N - GS; j += GS)
  {
#pragma HLS UNROLL factor = 4
    xs_buffer[j / GS] = xs[j / GS];
  }

  // 각 row(i)에 대해 matmul 수행
  for (int i = 0; i < D; i++)
  {
#pragma HLS PIPELINE
    float val = 0.0f;

    // 이 row의 가중치와 스케일을 임시로 읽어들임
    int8_t w_buffer[N];
    float ws_buffer[N / GS];

#pragma HLS ARRAY_PARTITION variable = w_buffer type = cyclic factor = 32
#pragma HLS ARRAY_PARTITION variable = ws_buffer type = cyclic factor = 32

    const int in = i * N;         // wq에서 i번째 row 시작
    const int in_s = i * N / GS;  // ws에서 i번째 row 시작
    const int groups = N / GS;    // 그룹 개수

  matmul1:
    for (int j = 0; j < N; j++)
    {
      w_buffer[j] = wq[j + in];
    }
  matmul2:
    for (int j = 0; j < groups; j++)
    {
      ws_buffer[j] = ws[in_s + j];
    }

    // 실제 곱셈 (GS 단위로 언롤링)
  matmul3:
    for (int j = 0; j <= N - GS; j += GS)
    {
      int32_t ival = 0;
    matmul4:
      for (int k = 0; k < GS; k++)
      {
        ival += ((int32_t)x_buffer[j + k]) * ((int32_t)w_buffer[j + k]);
      }
      // int32 누적값에 float 스케일 적용
      val += ((float)ival) * ws_buffer[j / GS] * xs_buffer[j / GS];
    }
    // 최종 결과
    xout[i] = val;
  }
}

//---------------------------------------------------------------------------------------
// 5) forward() 함수 (extern "C")
//    - Transformer 한 스텝(한 토큰) 추론을 수행하여 out(logits)을 계산
//    - HLS 커널로서 외부에서 호출되며, 다음 인자를 받는다:
//       * transformer: 모델 가중치(Transformer<...> 구조체)
//       * token: 현재 입력 토큰
//       * pos: 시퀀스 상 현재 위치
//       * key_cache, value_cache: K/V 캐시 배열
//       * out: 최종 로짓 저장 공간
//---------------------------------------------------------------------------------------
extern "C" void forward(
  Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *transformer,
  int token,
  int pos,
  float key_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)],
  float value_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)],
  float *out
)
{
#pragma HLS INTERFACE m_axi port = transformer offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem1

  // 간단히 사용할 변수들
  auto w = &transformer->weights;  // 가중치 구조체 포인터
  constexpr int UNROLL_FACTOR = 16;

  // 모델 내에서 쓸 임시 버퍼/배열
  // x : 현재 시점의 활성화(임베딩 + 레이어 쌓인 결과)
  static float x[config.dim];
  // xb, xb2 : 어텐션 및 FFN 중간 결과를 임시 저장
  static float xb[config.dim];
  static float xb2[config.dim];
  // hb, hb2 : FeedForward hidden_dim 중간 버퍼
  static float hb[config.hidden_dim];
  static float hb2[config.hidden_dim];
  // xq, hq : 양자화된 텐서(입력, hidden)
  static QuantizedTensor<config.dim> xq;
  static QuantizedTensor<config.hidden_dim> hq;
  // q, k, v : Self-Attention용 쿼리, 키, 값
  static float q[config.dim];
  static float k[(config.dim * config.n_kv_heads) / config.n_heads];
  static float v[(config.dim * config.n_kv_heads) / config.n_heads];
  // att : 어텐션 스코어(softmax 전/후) 저장
  static float att[config.n_heads * config.seq_len];

  // HLS 최적화 지시어(배열 파티셔닝, 언롤링 등)
#pragma HLS ARRAY_PARTITION variable = q cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = k cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = v cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = att cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = hq.q cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = hq.s cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = xq.q cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = xq.s cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = hb type = cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = hb2 type = cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = x type = cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = xb type = cyclic factor = UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = xb2 type = cyclic factor = UNROLL_FACTOR

  // kv_dim: Key/Value의 실제 벡터 크기 (Multi-Query 시 head 수가 다를 수 있음)
  constexpr int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
  constexpr int kv_mul = config.n_heads / config.n_kv_heads; // multiquery 시 헤드 공유 비율
  constexpr int head_size = dim / config.n_heads;            // 한 head당 차원 수

  // 1) token embedding 테이블에서 현재 토큰의 임베딩을 x에 복사
  std::memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));

  // 2) n_layers 만큼 반복 (모델의 레이어 스택)
main_forward_loop:
  for (int l = 0; l < config.n_layers; l++)
  {
    // (2-1) RMSNorm (Attention용)
    rmsnorm<dim>(xb, x, w->rms_att_weight + l * dim);

    // (2-2) Q, K, V를 계산 (matmul)
    //       먼저 xb를 양자화 -> matmul -> q/k/v
    quantize(&xq, xb, GS);
    matmul<dim, dim>(q, xq.q, xq.s, (w->wq + l)->q, (w->wq + l)->s);
    matmul<dim, kv_dim>(k, xq.q, xq.s, (w->wk + l)->q, (w->wk + l)->s);
    matmul<dim, kv_dim>(v, xq.q, xq.s, (w->wv + l)->q, (w->wv + l)->s);

    // (2-3) RoPE(회전 위치 인코딩): q, k를 head별로 (2개씩) 회전 변환
    //      첫번째 루프: i < kv_dim (Q, K 모두 적용)
  rotation1:
    for (int i = 0; i < kv_dim; i += 2)
    {
#pragma HLS UNROLL factor = UNROLL_FACTOR
#pragma HLS PIPELINE
      int head_dim = i % head_size;
      float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);

      // 쿼리 벡터 회전
      float v0_q = q[i];
      float v1_q = q[i + 1];
      q[i]     = v0_q * fcr - v1_q * fci;
      q[i + 1] = v0_q * fci + v1_q * fcr;

      // 키 벡터 회전
      float v0_k = k[i];
      float v1_k = k[i + 1];
      k[i]     = v0_k * fcr - v1_k * fci;
      k[i + 1] = v0_k * fci + v1_k * fcr;
    }

    //      두번째 루프: i >= kv_dim (Q에만 적용)
  rotation2:
    for (int i = kv_dim; i < dim; i += 2)
    {
#pragma HLS PIPELINE
      int head_dim = i % head_size;
      float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);

      float v0 = q[i];
      float v1 = q[i + 1];
      q[i]     = v0 * fcr - v1 * fci;
      q[i + 1] = v0 * fci + v1 * fcr;
    }

    // (2-4) 현재 pos 위치의 K, V를 key_cache, value_cache에 저장
    int loff = l * config.seq_len * kv_dim;   // 레이어별 offset
    float *key_cache_row   = key_cache   + loff + pos * kv_dim;
    float *value_cache_row = value_cache + loff + pos * kv_dim;
    std::memcpy(key_cache_row, k, kv_dim * sizeof(*key_cache_row));
    std::memcpy(value_cache_row, v, kv_dim * sizeof(*value_cache_row));

    // (2-5) Multi-head attention
  multihead_attention:
    for (int h = 0; h < n_heads; h++)
    {
      // h번째 head의 쿼리, 어텐션 스코어 배열 오프셋
      const int q_offset   = h * head_size;
      const int att_offset = h * seq_len;

    iterate:
      // 0~pos까지 과거 timestep 모두와 dot product -> att 스코어
      for (int t = 0; t <= pos; t++)
      {
#pragma HLS PIPELINE
#pragma HLS loop_tripcount min = 0 max = 257 avg = 129
        const int key_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
        float score = 0.0f;
        // dot product(q, key)
        for (int i = 0; i < head_size; i++)
        {
#pragma HLS unroll
          score += q[i + q_offset] * key_cache[i + key_offset];
        }
        // scale by sqrt(d)
        score /= sqrtf(head_size);
        att[t + att_offset] = score;
      }

      // (2-6) softmax로 attention weights 구하기
      softmax<257>(att + att_offset, pos + 1);

      // (2-7) weighted sum of values -> xb에 누적
      const int xb_offset = h * head_size;
      memset(xb + xb_offset, 0, head_size * sizeof(float));

    acc:
      for (int t = 0; t <= pos; t++)
      {
#pragma HLS loop_tripcount min = 0 max = 257 avg = 129
#pragma HLS PIPELINE
        const int v_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
        float a = att[t + att_offset];
      acc_inner:
        for (int i = 0; i < head_size; i++)
        {
#pragma HLS unroll
          xb[i + xb_offset] += a * value_cache[i + v_offset];
        }
      }
    }

    // (2-8) WO matmul
    quantize(&xq, xb, GS);
    matmul<dim, dim>(xb2, xq.q, xq.s, (w->wo + l)->q, (w->wo + l)->s);

    // (2-9) Residual connection: x += xb2
  residual:
    for (int i = 0; i < dim; i++)
    {
#pragma HLS UNROLL factor = 64 skip_exit_check
      x[i] += xb2[i];
    }

    // (2-10) FeedForward RMSNorm
    rmsnorm<dim>(xb, x, w->rms_ffn_weight + l * dim);

    // (2-11) FFN = w2(SiLU(w1(x)) * w3(x))
    quantize(&xq, xb, GS);
    matmul<dim, hidden_dim>(hb,  xq.q, xq.s, (w->w1 + l)->q, (w->w1 + l)->s);
    matmul<dim, hidden_dim>(hb2, xq.q, xq.s, (w->w3 + l)->q, (w->w3 + l)->s);

    float hb_out[hidden_dim];
#pragma HLS array_partition variable = hb_out type = cyclic factor = 16

  swi_glu:
    for (int i = 0; i < hidden_dim; i++)
    {
#pragma HLS UNROLL factor = 4
#pragma HLS PIPELINE
      // hb[i]에 SiLU 적용 => val = x * sigmoid(x)
      float val = hb[i];
      val *= (1.0f / (1.0f + expf(-val))); // SiLU(x) = x * sigmoid(x)
      // w3(x)와 elementwise 곱
      val *= hb2[i];
      hb_out[i] = val;
    }
    // hb_out -> hb
    std::memcpy(hb, hb_out, hidden_dim * sizeof(float));

    // (2-12) matmul w2
    quantize(&hq, hb, GS);
    matmul<hidden_dim, dim>(xb, hq.q, hq.s, (w->w2 + l)->q, (w->w2 + l)->s);

    // (2-13) Residual connection: x += xb
  residual2:
    for (int i = 0; i < dim; i++)
    {
#pragma HLS UNROLL factor = 16 skip_exit_check
      x[i] += xb[i];
    }
  }

  // (3) 모든 레이어를 거친 뒤 최종 RMSNorm
  rmsnorm<dim>(x, x, w->rms_final_weight);

  // (4) 분류기(출력 레이어) matmul -> logits 계산
  quantize(&xq, x, GS);
  matmul<dim, vocab_size>(out, xq.q, xq.s, w->wcls->q, w->wcls->s);
}
