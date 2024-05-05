#include "common.h"
#include "debug_utils.h"
#include "da.h"

const f32 training_data[] = {
    0, 0, 1, //
    1, 0, 1, //
    0, 1, 1, //
    1, 1, 0, //
};

typedef struct NN {
  f32 w1;
  f32 w2;
  f32 bias;
} NN;

f32 sigmoidf(f32 x) {
  return 1 / (1 + expf(-x));
}

f32 forward(NN nn, f32 x1, f32 x2) {
  return sigmoidf(nn.w1 * x1 + nn.w2 * x2 + nn.bias);
}

/// `training_input_size` is number of floats, not number of input-outputs.
/// Returns the loss.
f32 train(NN *nn, const f32 *training_input, usize training_input_size, usize i, f32 rate) {
  assert(training_input_size % 2 == 0);
  f32 loss = 0;
  f32 dw1 = 0;
  f32 dw2 = 0;
  f32 db = 0;
  for (usize i = 0; i < training_input_size; i += 3) {
    const f32 x1 = training_input[i];
    const f32 x2 = training_input[i + 1];
    const f32 y = training_input[i + 2];
    f32 nn_out = forward(*nn, x1, x2);
    f32 diff = nn_out - y;
    loss += diff * diff;
    dw1 += diff * nn_out * (1 - nn_out) * x1;
    dw2 += diff * nn_out * (1 - nn_out) * x2;
    db += diff * nn_out * (1 - nn_out);
  }
  usize n = training_input_size / 3;
  loss = loss / n;
  dw1 = dw1 / n * 2;
  dw2 = dw2 / n * 2;
  db = db / n * 2;
  if ((i % 1000) == 0)
    printf("%zu\tloss: %.08f\n", i, loss);
  nn->bias -= db * rate;
  nn->w1 -= dw1 * rate;
  nn->w2 -= dw2 * rate;
  return loss;
}

f32 randf0to1() {
  return (f32)rand() / (f32)RAND_MAX;
}

/// Arguments: ceil >= floor
f32 randf_in(f32 floor, f32 ceil) {
  ASSERT(ceil >= floor);
  return floor + (ceil - floor) * randf0to1();
}

NN neuron_random(f32 floor, f32 ceil) {
  return (NN){
      .w1 = randf_in(floor, ceil),
      .bias = randf_in(floor, ceil),
  };
}

int main() {
  srand(time(NULL));
  NN nn = neuron_random(-1, 1);
  DBG_PRINTLN(nn.w1);
  DBG_PRINTLN(nn.bias);
  usize training_rounds = 100 * 1000;
  for (usize i = 0; i < training_rounds; ++i) {
    train(&nn, training_data, ARR_LEN(training_data), i, 1e-2);
  }
  printf("-------------------------\n");
  DBG_PRINTLN(nn.w1);
  DBG_PRINTLN(nn.bias);
  for (usize i = 0; i < ARR_LEN(training_data); i += 3) {
    f32 x1 = training_data[i];
    f32 x2 = training_data[i + 1];
    f32 out = forward(nn, x1, x2);
    printf("%.0f, %.0f => %.04f ~ %.0f\n", x1, x2, out, roundf(out));
  }
  return 0;
}
