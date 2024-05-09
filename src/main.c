#include "common.h"
#include "debug_utils.h"
#include "da.h"

DECL_DA_STRUCT(f32, DynArrayF32);

typedef struct SliceF32 {
  f32 *values;
  usize len;
} SliceF32;

f32 randf_0to1() {
  return (f32)rand() / (f32)RAND_MAX;
}

/// Arguments: ceil >= floor
f32 randf_in(f32 floor, f32 ceil) {
  DEBUG_ASSERT(ceil >= floor);
  return floor + (ceil - floor) * randf_0to1();
}

f32 sigmoidf(f32 x) {
  return 1 / (1 + expf(-x));
}

/// Tensor is a cringe name,
/// I'm gonna call it Mat, short for Mattress.
/// Does not own the data.
typedef struct Mat {
  f32 *values;
  usize cols;
  usize rows;
} Mat;

/// Perform sigmoid on every element of a matrix.
void sigmoid_mat(Mat m) {
  for (usize i = 0; i < m.rows * m.cols; ++i) {
    f32 *x = &m.values[i];
    *x = sigmoidf(*x);
  }
}

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
  /// A pool of floats.
  DynArrayF32 pool;
  DynArrayMat ws;
  DynArrayMat bs;
  DynArrayMat as;
} NN;

/// The first layer is the number of inputs.
/// Must have at least 2 layers (0th layer for input and 1 layer of neurons).
NN nn_new(usize *layers, usize layers_count) {
  ASSERT(layers_count > 1);
  DynArrayMat ws = {0};
  DynArrayMat bs = {0};
  DynArrayMat as = {0};
  DynArrayF32 pool = {0};
  da_reserve_exact(&ws, layers_count - 1);
  da_reserve_exact(&bs, layers_count - 1);
  da_reserve_exact(&as, layers_count - 1);
  for (usize i = 1; i < layers_count; ++i) {
    usize layer = layers[i];
    usize prev_layer = layers[i - 1];
    usize idx = pool.da_len;
    da_append_zeros(&pool, prev_layer * layer + layer + layer);
    da_push(&ws, ((Mat){
                     .cols = prev_layer,
                     .rows = layer,
                     .values = da_get(&pool, idx),
                 }));
    idx += prev_layer * layer;
    da_push(&bs, ((Mat){
                     .cols = 1,
                     .rows = layer,
                     .values = da_get(&pool, idx),
                 }));
    idx += layer;
    da_push(&as, ((Mat){
                     .cols = 1,
                     .rows = layer,
                     .values = da_get(&pool, idx),
                 }));
  }
  return (NN){
      .pool = pool,
      .ws = ws,
      .bs = bs,
      .as = as,
  };
}

void nn_free(NN nn) {
  da_free(nn.pool);
  da_free(nn.ws);
  da_free(nn.bs);
  da_free(nn.as);
}

/// Not including input layer.
usize nn_layer_count(NN nn) {
  return nn.as.da_len;
}

SliceF32 nn_forward(NN nn, SliceF32 input) {
  Mat a0 = {
      .cols = 1,
      .rows = input.len,
      .values = input.values,
  };
  for (usize l = 0; l < nn_layer_count(nn); ++l) {
    Mat a_ = l == 0 ? a0 : *da_get(&nn.as, l - 1); // a previous layer
    Mat a = *da_get(&nn.as, l);
    Mat w = *da_get(&nn.ws, l);
    Mat b = *da_get(&nn.bs, l);
    mat_mul(a, w, a_);
    mat_add(a, b);
    sigmoid_mat(a);
  }
  Mat out = *da_get(&nn.as, nn.as.da_len - 1);
  return (SliceF32){out.values, out.cols};
}

f32 nn_train(NN *nn, f32 *training_input, usize training_input_size, f32 rate, usize i) {
  if (nn->as.da_len != 1)
    TODO();
  if (da_get(&nn->as, 0)->rows != 1)
    TODO();
  if (da_get(&nn->ws, 0)->cols != 2)
    TODO();
  assert(training_input_size % 2 == 0);
  f32 loss = 0;
  f32 dw1 = 0;
  f32 dw2 = 0;
  f32 db = 0;
  for (usize i = 0; i < training_input_size; i += 3) {
    f32 *in = &training_input[i];
    const f32 y = training_input[i + 2];
    f32 nn_out = nn_forward(*nn, (SliceF32){in, 2}).values[0];
    f32 diff = nn_out - y;
    loss += diff * diff;
    dw1 += diff * nn_out * (1 - nn_out) * in[0];
    dw2 += diff * nn_out * (1 - nn_out) * in[1];
    db += diff * nn_out * (1 - nn_out);
  }
  usize n = training_input_size / 3;
  loss = loss / n;
  dw1 = dw1 / n * 2;
  dw2 = dw2 / n * 2;
  db = db / n * 2;
  if ((i % 1000) == 0)
    printf("%zu\tloss: %.08f\n", i, loss);
  da_get(&nn->bs, 0)->values[0] -= db * rate;
  da_get(&nn->ws, 0)->values[0] -= dw1 * rate;
  da_get(&nn->ws, 0)->values[1] -= dw2 * rate;
  return loss;
}

// AND gate.
f32 training_data[] = {
    0, 0, 1, //
    1, 0, 1, //
    0, 1, 1, //
    1, 1, 0, //
};

int main() {
  usize layers[] = {2, 1};
  NN nn = nn_new(layers, ARR_LEN(layers));

  // Print matrices in the network.
  for (usize i = 0; i < ARR_LEN(layers) - 1; ++i) {
    DBG_PRINTLN(i);
    DBG_PRINTF("ws = \n");
    mat_println(*da_get(&nn.ws, i));
    DBG_PRINTF("bs = \n");
    mat_println(*da_get(&nn.bs, i));
    DBG_PRINTF("as = \n");
    mat_println(*da_get(&nn.as, i));
  }

  usize training_rounds = 100 * 1000;
  for (usize i = 0; i < training_rounds; ++i) {
    nn_train(&nn, training_data, ARR_LEN(training_data), 1, i);
  }

  for (usize i = 0; i < ARR_LEN(training_data); i += 3) {
    SliceF32 out = nn_forward(nn, (SliceF32){&training_data[i], 2});
    printf("%.0f, %.0f => %.04f ~ %.0f\n", training_data[i], training_data[i + 1], out.values[0],
           roundf(out.values[0]));
  }

  nn_free(nn);

  return 0;
}
