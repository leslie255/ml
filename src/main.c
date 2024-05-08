#include "common.h"
#include "debug_utils.h"
#include "da.h"

DECL_DA_STRUCT(f32, DynArrayF32);

f32 randf_0to1() {
  return (f32)rand() / (f32)RAND_MAX;
}

/// Arguments: ceil >= floor
f32 randf_in(f32 floor, f32 ceil) {
  DEBUG_ASSERT(ceil >= floor);
  return floor + (ceil - floor) * randf_0to1();
}

/// Tensor is a cringe name,
/// I'm gonna call it Mat, short for Mattress.
/// Does not own the data.
typedef struct Mat {
  f32 *values;
  usize cols;
  usize rows;
} Mat;

DECL_DA_STRUCT(Mat, DynArrayMat);

f32 *mat_get(Mat m, usize x, usize y) {
  return &m.values[y * m.cols + x];
}

/// SAFETY: Data of dest must not overlap with either of lhs or rhs.
void mat_mul(Mat dest, Mat lhs, Mat rhs) {
  // (4x3) * (3x4)
  DEBUG_ASSERT(lhs.cols == rhs.rows);
  DEBUG_ASSERT(dest.rows == lhs.rows);
  DEBUG_ASSERT(dest.cols == rhs.cols);
  for (usize y = 0; y < dest.rows; ++y) {
    for (usize x = 0; x < dest.cols; ++x) {
      f32 *dest_val = mat_get(dest, x, y);
      *dest_val = 0;
      for (usize i = 0; i < lhs.cols; ++i) {
        *dest_val += *mat_get(lhs, i, y) * *mat_get(rhs, x, i);
      }
    }
  }
}

void mat_add(Mat dest, Mat rhs) {
  DEBUG_ASSERT(dest.cols == rhs.cols);
  DEBUG_ASSERT(dest.rows == rhs.rows);
  for (usize y = 0; y < dest.rows; ++y) {
    for (usize x = 0; x < dest.cols; ++x) {
      *mat_get(dest, x, y) += *mat_get(rhs, x, y);
    }
  }
}

void mat_println(Mat m) {
  for (usize y = 0; y < m.rows; ++y) {
    printf("[ ");
    for (usize x = 0; x < m.cols; ++x) {
      i32 len = printf("%.3f", *mat_get(m, x, y));
      if (len < 8)
        printf("%*s", 8 - len, "");
      else if (len >= 8)
        printf(" ");
      if (x == m.cols - 1)
        printf(" ]\n");
    }
  }
}

void mat_rand(Mat m, f32 floor, f32 ceil) {
  for (usize y = 0; y < m.rows; ++y) {
    for (usize x = 0; x < m.cols; ++x) {
      *mat_get(m, x, y) = randf_in(floor, ceil);
    }
  }
}

typedef struct NN {
  DynArrayF32 pool;
  DynArrayMat ws;
  DynArrayMat bs;
  DynArrayMat as;
} NN;

NN nn_new(usize *layers, usize layers_count) {
  (void)layers;
  DynArrayMat ws = {0};
  DynArrayMat bs = {0};
  DynArrayMat as = {0};
  da_reserve(&ws, layers_count);
  da_reserve(&bs, layers_count);
  da_reserve(&as, layers_count);
  for (usize i = 0; i < layers_count; ++i) {
  }
  TODO_FUNCTION();
}

void nn_free(NN nn) {
  da_free(nn.pool);
}

typedef struct NN_ {
  f32 w1;
  f32 w2;
  f32 bias;
} NN_;

f32 sigmoidf(f32 x) {
  return 1 / (1 + expf(-x));
}

f32 forward(NN_ nn, f32 x1, f32 x2) {
  return sigmoidf(nn.w1 * x1 + nn.w2 * x2 + nn.bias);
}

/// `training_input_size` is number of floats, not number of input-outputs.
/// Returns the loss.
f32 train(NN_ *nn, const f32 *training_input, usize training_input_size, usize i, f32 rate) {
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

NN_ neuron_random(f32 floor, f32 ceil) {
  return (NN_){
      .w1 = randf_in(floor, ceil),
      .bias = randf_in(floor, ceil),
  };
}

// AND gate.
const f32 training_data[] = {
    0, 0, 1, //
    1, 0, 1, //
    0, 1, 1, //
    1, 1, 0, //
};

int main() {
  srand(255);
  f32 values1[12] = {
      2, 3,  4,  5,   //
      1, 10, 89, 20,  //
      6, 7,  11, 123, //
  };
  Mat m1 = {
      .cols = 4,
      .rows = 3,
      .values = values1,
  };
  DBG_PRINTF("m1 =\n");
  mat_println(m1);
  f32 values2[12] = {
      1, 2, 2, //
      1, 3, 1, //
      2, 2, 1, //
      1, 1, 2, //
  };
  Mat m2 = {
      .cols = 3,
      .rows = 4,
      .values = values2,
  };
  DBG_PRINTF("m2 =\n");
  mat_println(m2);
  f32 values3[9] = {0};
  Mat m3 = {
      .cols = 3,
      .rows = 3,
      .values = values3,
  };
  mat_mul(m3, m1, m2);
  DBG_PRINTF("m1 * m2 =\n");
  mat_println(m3);

  // srand(time(NULL));
  // NN_ nn = neuron_random(-1, 1);
  // DBG_PRINTLN(nn.w1);
  // DBG_PRINTLN(nn.w2);
  // DBG_PRINTLN(nn.bias);
  // usize training_rounds = 100 * 1000;
  // for (usize i = 0; i < training_rounds; ++i) {
  //   train(&nn, training_data, ARR_LEN(training_data), i, 1);
  // }
  // printf("-------------------------\n");
  // DBG_PRINTLN(nn.w1);
  // DBG_PRINTLN(nn.w2);
  // DBG_PRINTLN(nn.bias);
  // for (usize i = 0; i < ARR_LEN(training_data); i += 3) {
  //   f32 x1 = training_data[i];
  //   f32 x2 = training_data[i + 1];
  //   f32 out = forward(nn, x1, x2);
  //   printf("%.0f, %.0f => %.04f ~ %.0f\n", x1, x2, out, roundf(out));
  // }
  return 0;
}
