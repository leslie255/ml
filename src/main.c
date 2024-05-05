#include "common.h"
#include "debug_utils.h"
#include "da.h"

const f32 training_data[] = {
    0, 1, //
    1, 0, //
};

typedef struct NN {
  f32 weight;
  f32 bias;
} NN;

f32 sigmoidf(f32 x) {
  return 1 / (1 + expf(-x));
}

f32 forward(NN nn, f32 in) {
  return sigmoidf(nn.weight * in + nn.bias);
}

/// `training_count` is number of items (not number of pairs!)
/// Returns the loss.
f32 train(NN *nn, const f32 *training_pairs, usize training_count, usize i, f32 rate) {
  assert(training_count % 2 == 0);
  f32 loss = 0;
  f32 dw = 0;
  f32 db = 0;
  for (usize i = 0; i < training_count; i += 2) {
    const f32 training_in = training_pairs[i];
    const f32 training_out = training_pairs[i + 1];
    f32 nn_out = forward(*nn, training_in);
    f32 diff = nn_out - training_out;
    loss += diff * diff;
    dw += diff * nn_out * (1 - nn_out) * training_in;
    db += diff * nn_out * (1 - nn_out);
  }
  usize n = training_count / 2;
  loss = loss / n;
  dw = dw / n * 2;
  db = db / n * 2;
  if ((i % 1000) == 0)
    printf("%zu\tloss: %.08f\tw: %.04f\tdw: %.04f\tb: %.04f\tdb: %.04f\n", i, loss, nn->weight, dw, nn->bias, db);
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
  srand(time(NULL));
  NN nn = neuron_random(-1, 1);
  DBG_PRINTLN(nn.weight);
  DBG_PRINTLN(nn.bias);
  usize training_rounds = 100 * 1000;
  for (usize i = 0; i < training_rounds; ++i) {
    train(&nn, training_data, ARR_LEN(training_data), i, 1e-2);
  }
  printf("-------------------------\n");
  DBG_PRINTLN(nn.weight);
  DBG_PRINTLN(nn.bias);
  for (usize i = 0; i < ARR_LEN(training_data); i += 2) {
    f32 in = training_data[i];
    f32 out = forward(nn, in);
    printf("%.0f => %.04f\n", in, out);
  }
  return 0;
}
