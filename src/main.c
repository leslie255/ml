#include "common.h"
#include "debug_utils.h"
#include "da.h"

DECL_DA_STRUCT(f32, DynArrayF32);
DECL_SLICE_STRUCT(f32, SliceF32);

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

/// Tensor is a cringe name,
/// I'm gonna call it Mat, short for Mattress.
/// Does not own the data.
typedef struct ConstMat {
  const f32 *values;
  usize cols;
  usize rows;
} ConstMat;

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

const f32 *mat_get_(ConstMat m, usize x, usize y) {
  return &m.values[y * m.cols + x];
}

attribute(const, always_inline) ConstMat mat_as_const(Mat m) {
  return PTR_CAST(ConstMat, m);
}

/// SAFETY: Data of dest must not overlap with either of lhs or rhs.
void mat_mul(Mat dest, ConstMat lhs, ConstMat rhs) {
  // (4x3) * (3x4)
  DEBUG_ASSERT(lhs.cols == rhs.rows);
  DEBUG_ASSERT(dest.rows == lhs.rows);
  DEBUG_ASSERT(dest.cols == rhs.cols);
  for (usize y = 0; y < dest.rows; ++y) {
    for (usize x = 0; x < dest.cols; ++x) {
      f32 *dest_val = mat_get(dest, x, y);
      *dest_val = 0;
      for (usize i = 0; i < lhs.cols; ++i) {
        *dest_val += *mat_get_(lhs, i, y) * *mat_get_(rhs, x, i);
      }
    }
  }
}

void mat_add(Mat dest, ConstMat rhs) {
  DEBUG_ASSERT(dest.cols == rhs.cols);
  DEBUG_ASSERT(dest.rows == rhs.rows);
  for (usize y = 0; y < dest.rows; ++y) {
    for (usize x = 0; x < dest.cols; ++x) {
      *mat_get(dest, x, y) += *mat_get_(rhs, x, y);
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

/// Number of inputs of a neuron network.
usize nn_input_count(NN nn) {
  return da_get(&nn.ws, 0)->cols;
}

/// Number of neuron in layer in a neural network.
/// `layer` does not include inputs.
usize nn_neuron_count_in_layer(NN nn, usize layer) {
  return da_get(&nn.as, layer)->rows;
}

/// Number of inputs of a neuron network.
usize nn_output_count(NN nn) {
  return nn_neuron_count_in_layer(nn, nn_layer_count(nn) - 1);
}

/// SAFETY: `input` must be an array of same number of elements as input layer.
/// Returns reference to the last layer (output layer).
const f32 *nn_forward(NN nn, const f32 *input) {
  ConstMat a0 = {
      .cols = 1,
      .rows = nn_input_count(nn),
      .values = input,
  };
  for (usize l = 0; l < nn_layer_count(nn); ++l) {
    ConstMat a_ = l == 0 ? a0 : mat_as_const(*da_get(&nn.as, l - 1)); // a previous layer
    Mat a = *da_get(&nn.as, l);
    ConstMat w = mat_as_const(*da_get(&nn.ws, l));
    ConstMat b = mat_as_const(*da_get(&nn.bs, l));
    mat_mul(a, w, a_);
    mat_add(a, b);
    sigmoid_mat(a);
  }
  Mat out = *da_get(&nn.as, nn.as.da_len - 1);
  return out.values;
}

void da_free_f32(DynArrayF32 *da) {
  da_free(*da);
}

/// `i` is the current round of training.
/// It's only used for debug logging, leave zero if not needed.
f32 nn_train(NN *nn, f32 *training_input, usize training_input_size, f32 rate, usize i) {
  (void)rate;
  // Only handles 1 neuron per layer rn.
  if (nn_input_count(*nn) != 1)
    PANIC_PRINTF("TODO: more than 1 inputs\n");
  for (usize l = 0; l < nn_layer_count(*nn); ++l)
    if (nn_neuron_count_in_layer(*nn, l) != 1)
      PANIC_PRINTF("TODO: more than 1 neurons per layer\n");

  const usize stride = nn_input_count(*nn) + nn_output_count(*nn);
  const usize n = training_input_size / stride;
  const usize m = nn_layer_count(*nn);
  assert(training_input_size % stride == 0);

  // TODO: In future maybe store this into a `TrainingContext` struct to reduce allocations.
  f32 *dws = xalloc(f32, m);
  f32 *dbs = xalloc(f32, m);

  f32 loss = 0;
  f32 prod = 1;
  for (usize i = 0; i < training_input_size; i += stride) {
    f32 *a0 = &training_input[i];
    f32 *y = &training_input[i + nn_output_count(*nn)];
    f32 am = *nn_forward(*nn, a0);
    f32 diff = am - *y;
    loss += diff * diff;

    DBG_PRINTLN(am);
    for (usize j = m - 1; j != SIZE_MAX; --j) {
      DBG_PRINTLN(j);
      if (j != m - 1) {
        f32 a_j = da_get(&nn->as, j)->values[j];
        f32 w_j = da_get(&nn->ws, j)->values[j];
        f32 daa = a_j * (1 - a_j) * w_j;
        prod *= daa;
      }
      f32 daL = 2 / (f32)n * diff * prod;
      f32 a_j = da_get(&nn->as, j)->values[0];
      f32 a_prev = j > 0 ? da_get(&nn->as, j - 1)->values[0] : *a0;
      dws[j] += daL * a_j * (1 - a_j) * a_prev;
      dbs[j] += daL * a_j * (1 - a_j);
    }
  }
  printf("%zu", i);
  for (usize k = 0; k < m; ++k) {
    dws[k] /= n;
    dbs[k] /= n;
    printf("\tdw%zu=%.4f\tdb%zu=%.4f\n", k + 1, dws[k], k + 1, dbs[k]);
    da_get(&nn->ws, k)->values[0] -= dws[k];
    da_get(&nn->bs, k)->values[0] -= dbs[k];
  }
  loss /= n;
  printf("loss: %.08f\n", loss);

  xfree(dws);
  xfree(dbs);

  return loss;
}

// AND gate.
f32 training_data[] = {
    1, 0, //
    0, 1, //
};

int main() {
  usize layers[] = {1, 1};
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

  usize training_rounds = 1000;
  for (usize i = 0; i < training_rounds; ++i) {
    nn_train(&nn, training_data, ARR_LEN(training_data), 1e-2, i);
  }

  for (usize i = 0; i < ARR_LEN(training_data); i += nn_input_count(nn) + nn_output_count(nn)) {
    f32 out = *nn_forward(nn, &training_data[i]);
    printf("%.0f => %.04f ~ %.0f\n", training_data[i], out, roundf(out));
  }

  // Print matrices in the network.
  printf("--------------------------------\n");
  for (usize i = 0; i < ARR_LEN(layers) - 1; ++i) {
    DBG_PRINTLN(i);
    DBG_PRINTF("ws = \n");
    mat_println(*da_get(&nn.ws, i));
    DBG_PRINTF("bs = \n");
    mat_println(*da_get(&nn.bs, i));
    DBG_PRINTF("as = \n");
    mat_println(*da_get(&nn.as, i));
  }

  nn_free(nn);

  return 0;
}
