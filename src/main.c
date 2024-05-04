#include "common.h"
#include "debug_utils.h"
#include "da.h"

const f32 training_data[] = {
    0,  0 * -2,  //
    1,  1 * -2,  //
    2,  2 * -2,  //
    3,  3 * -2,  //
    4,  4 * -2,  //
    5,  5 * -2,  //
    6,  6 * -2,  //
    7,  7 * -2,  //
    8,  8 * -2,  //
    9,  9 * -2,  //
    10, 10 * -2, //
    11, 11 * -2, //
    12, 12 * -2, //
    13, 13 * -2, //
    14, 14 * -2, //
    15, 15 * -2, //
    16, 16 * -2, //
    17, 17 * -2, //
    18, 18 * -2, //
};

typedef struct NN {
  f32 weight;
  f32 bias;
} NN;

f32 forward(NN nn, f32 in) {
  return nn.weight * in + nn.bias;
}

/// `training_count` is number of items (not number of pairs!)
/// Returns the loss.
f32 train(NN *nn, const f32 *training_pairs, usize training_count, usize i, f32 rate) {
  assert(training_count % 2 == 0);
  f32 sumpow2 = 0;
  f32 sum = 0;
  for (usize i = 0; i < training_count; i += 2) {
    const f32 training_in = training_pairs[i];
    const f32 training_out = training_pairs[i + 1];
    f32 nn_out = forward(*nn, training_in);
    f32 diff = nn_out - training_out;
    sumpow2 += diff * diff;
    sum += diff;
  }
  f32 loss = sumpow2 / training_count;
  f32 dw = sum / training_count * 2;
  f32 db = sum / training_count * 2;
  if ((i % (1000 / 10)) == 0)
    printf("%zu\tloss: %.04f\tw: %.04f\tdw: %.04f\tb: %.04f\tdb: %.04f\n", i, loss, nn->weight, dw, nn->bias, db);
  nn->bias -= db * rate;
  nn->weight -= dw * rate;
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
      .weight = randf_in(floor, ceil),
      .bias = randf_in(floor, ceil),
  };
}

int main() {
  srand(10);
  NN nn = neuron_random(-5, 5);
  DBG_PRINTLN(nn.weight);
  DBG_PRINTLN(nn.bias);
  usize training_rounds = 1000;
  for (usize i = 0; i < training_rounds; ++i) {
    train(&nn, training_data, ARR_LEN(training_data), i, 1e-3);
  }
  printf("-------------------------\n");
  DBG_PRINTLN(nn.weight);
  DBG_PRINTLN(nn.bias);
  for (usize i = 0; i < ARR_LEN(training_data); i += 2) {
    f32 in = training_data[i];
    f32 out = forward(nn, in);
    printf("%f * -2 = %.04f ~= %.0f\n", in, out, roundf(out));
  }
  return 0;
}
