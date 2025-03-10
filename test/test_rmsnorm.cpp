#include <iostream>
#include <cmath>
#include <cstring>

// rmsnorm 함수 정의 (이미 작성된 코드 복사)
// 여기서는 HLS pragma는 컴파일러에 따라 무시될 수 있으므로, 일반 컴파일 시에도 문제없이 작동합니다.
template <int S>
void rmsnorm(float o[S], float x[S], float weight[S])
{
    constexpr auto array_size = S * sizeof(float);
    float ss = 0.0f;
    float x_buff[S];
    float weight_buff[S];
    float out_buff[S];

    // 배열 복사
    std::memcpy(x_buff, x, array_size);
    std::memcpy(weight_buff, weight, array_size);

sum_of_squares:
    for (int j = 0; j < S; j++)
    {
        float x_j = x_buff[j];
        ss += x_j * x_j;  // 제곱합 누적
    }

    ss /= S;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);

norm_and_scale:
    for (int j = 0; j < S; j++)
    {
        float weight_j = weight_buff[j];
        float x_j = x_buff[j];
        out_buff[j] = weight_j * (ss * x_j);
    }

    std::memcpy(o, out_buff, array_size);
}

int main() {
    constexpr int S = 8;  // 원하는 배열 크기, 여기서는 8
    float x[S] = {1.0f, 2.0f, -1.0f, 3.0f, -2.5f, 0.5f, 10.0f, 2.0f};
    float weight[S] = {1.0f, 1.1f, 0.9f, 1.2f, 0.8f, 1.05f, 0.95f, 1.0f};
    float o[S];  // 결과를 담을 배열

    // rmsnorm 호출
    rmsnorm<S>(o, x, weight);

    // 결과 출력
    std::cout << "RMSNorm output:\n";
    for (int i = 0; i < S; i++) {
        std::cout << "o[" << i << "] = " << o[i] << std::endl;
    }

    return 0;
}
