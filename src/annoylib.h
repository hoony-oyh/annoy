// Copyright (c) 2013 Spotify AB
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.


#ifndef ANNOYLIB_H
#define ANNOYLIB_H

#include <stdio.h>
#include <sys/stat.h>
#ifndef _MSC_VER
#include <unistd.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stddef.h>

#if defined(_MSC_VER) && _MSC_VER == 1500
typedef unsigned char     uint8_t;
typedef signed __int32    int32_t;
typedef unsigned __int64  uint64_t;
#else
#include <stdint.h>
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
 #ifndef NOMINMAX
  #define NOMINMAX
 #endif
 #include "mman.h"
 #include <windows.h>
#else
 #include <sys/mman.h>
#endif

#include <cerrno>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <queue>
#include <limits>

#ifdef _MSC_VER
// Needed for Visual Studio to disable runtime checks for mempcy
#pragma runtime_checks("s", off)
#endif

// This allows others to supply their own logger / error printer without
// requiring Annoy to import their headers. See RcppAnnoy for a use case.
#ifndef __ERROR_PRINTER_OVERRIDE__
  #define showUpdate(...) { fprintf(stderr, __VA_ARGS__ ); }
#else
  #define showUpdate(...) { __ERROR_PRINTER_OVERRIDE__( __VA_ARGS__ ); }
#endif


#ifndef _MSC_VER
#define popcount __builtin_popcountll
#else // See #293, #358
#define isnan(x) _isnan(x)
#define popcount cole_popcount
#endif

#if !defined(NO_MANUAL_VECTORIZATION) && defined(__GNUC__) && (__GNUC__ >6) && defined(__AVX512F__)  // See #402
#pragma message "Using 512-bit AVX instructions"
#define USE_AVX512
#elif !defined(NO_MANUAL_VECTORIZATION) && defined(__AVX__) && defined (__SSE__) && defined(__SSE2__) && defined(__SSE3__)
#pragma message "Using 256-bit AVX instructions"
#define USE_AVX
#else
#pragma message "Using no AVX instructions"
#endif

#if defined(USE_AVX) || defined(USE_AVX512)
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__)
#include <x86intrin.h>
#endif
#endif

#ifndef ANNOY_NODE_ATTRIBUTE
    #ifndef _MSC_VER
        #define ANNOY_NODE_ATTRIBUTE __attribute__((__packed__))
        // TODO: this is turned on by default, but may not work for all architectures! Need to investigate.
    #else
        #define ANNOY_NODE_ATTRIBUTE
    #endif
#endif


using std::vector;
using std::pair;
using std::numeric_limits;
using std::make_pair;

inline void* remap_memory(void* _ptr, int _fd, size_t old_size, size_t new_size) {
#ifdef __linux__
  _ptr = mremap(_ptr, old_size, new_size, MREMAP_MAYMOVE);
#else
  munmap(_ptr, old_size);
#ifdef MAP_POPULATE
  _ptr = mmap(_ptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, _fd, 0);
#else
  _ptr = mmap(_ptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, _fd, 0);
#endif
#endif
  return _ptr;
}

namespace {

template<typename S, typename Node>
inline Node* get_node_ptr(const void* _nodes, const size_t _s, const S i) {
  return (Node*)((uint8_t *)_nodes + (_s * i));
}

template<typename T>
inline T dot(const T* x, const T* y, int f) {
  T s = 0;
  for (int z = 0; z < f; z++) {
    s += (*x) * (*y);
    x++;
    y++;
  }
  return s;
}

template<typename T>
inline void substract(const T* x, const T* y, T* z, int f) {
    for (int i = 0; i < f; i++) {
        *z = (*x) - (*y);
        x++;
        y++;
        z++;
    }
}

template<typename T>
inline void scale(const T* x, T* z, T scale, int f) {
    for (int i = 0; i < f; i++) {
        *z = (*x) * scale;
        x++;
        z++;
    }
}

template<typename T>
inline void aXplusbY(const T* x, const T* y, T* z, T a, T b, int f) {
    for (int i = 0; i < f; i++) {
        *z = a * (*x) + b * (*y);
        x++;
        y++;
        z++;
    }
}

template<typename T>
inline T euclidean_distance(const T* x, const T* y, int f) {
  // Don't use dot-product: avoid catastrophic cancellation in #314.
  T d = 0.0;
  for (int i = 0; i < f; ++i) {
    const T tmp=*x - *y;
    d += tmp * tmp;
    ++x;
    ++y;
  }
  return d;
}

#ifdef USE_AVX
#include "kernel.h"

// TODO. implement better euclidean_distances
template<>
inline float euclidean_distance<float>(const float* x, const float* y, int f) {
  float result=0;
  if (f > 7) {
    __m256 d = _mm256_setzero_ps();
    for (; f > 7; f -= 8) {
      const __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y));
      d = _mm256_add_ps(d, _mm256_mul_ps(diff, diff)); // no support for fmadd in AVX...
      x += 8;
      y += 8;
    }
    // Sum all floats in dot register.
    result = hsum256_ps_avx(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    float tmp = *x - *y;
    result += tmp * tmp;
    x++;
    y++;
  }
  return result;
}

#endif

#ifdef USE_AVX512
template<>
inline float dot<float>(const float* x, const float *y, int f) {
  float result = 0;
  if (f > 15) {
    __m512 d = _mm512_setzero_ps();
    for (; f > 15; f -= 16) {
      //AVX512F includes FMA
      d = _mm512_fmadd_ps(_mm512_loadu_ps(x), _mm512_loadu_ps(y), d);
      x += 16;
      y += 16;
    }
    // Sum all floats in dot register.
    result += _mm512_reduce_add_ps(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    result += *x * *y;
    x++;
    y++;
  }
  return result;
}

template<>
inline float manhattan_distance<float>(const float* x, const float* y, int f) {
  float result = 0;
  int i = f;
  if (f > 15) {
    __m512 manhattan = _mm512_setzero_ps();
    for (; i > 15; i -= 16) {
      const __m512 x_minus_y = _mm512_sub_ps(_mm512_loadu_ps(x), _mm512_loadu_ps(y));
      manhattan = _mm512_add_ps(manhattan, _mm512_abs_ps(x_minus_y));
      x += 16;
      y += 16;
    }
    // Sum all floats in manhattan register.
    result = _mm512_reduce_add_ps(manhattan);
  }
  // Don't forget the remaining values.
  for (; i > 0; i--) {
    result += fabsf(*x - *y);
    x++;
    y++;
  }
  return result;
}

template<>
inline float euclidean_distance<float>(const float* x, const float* y, int f) {
  float result=0;
  if (f > 15) {
    __m512 d = _mm512_setzero_ps();
    for (; f > 15; f -= 16) {
      const __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(x), _mm512_loadu_ps(y));
      d = _mm512_fmadd_ps(diff, diff, d);
      x += 16;
      y += 16;
    }
    // Sum all floats in dot register.
    result = _mm512_reduce_add_ps(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    float tmp = *x - *y;
    result += tmp * tmp;
    x++;
    y++;
  }
  return result;
}

#endif

 
template<typename T>
inline T get_norm(T* v, int f) {
  return sqrt(dot(v, v, f));
}

template<typename T, typename Random, typename Distance, typename Node>
inline void two_means(const vector<Node*>& nodes, int f, Random& random, bool cosine, Node* p, Node* q) {
  /*
    This algorithm is a huge heuristic. Empirically it works really well, but I
    can't motivate it well. The basic idea is to keep two centroids and assign
    points to either one of them. We weight each centroid by the number of points
    assigned to it, so to balance it. 
  */
  static int iteration_steps = 200;
  size_t count = nodes.size();

  size_t i = random.index(count);
  size_t j = random.index(count-1);
  j += (j >= i); // ensure that i != j

  Distance::template copy_node<T, Node>(p, nodes[i], f);
  Distance::template copy_node<T, Node>(q, nodes[j], f);

  if (cosine) { Distance::template normalize<T, Node>(p, f); Distance::template normalize<T, Node>(q, f); }
  Distance::init_node(p, f);
  Distance::init_node(q, f);

  int ic = 1, jc = 1;
  for (int l = 0; l < iteration_steps; l++) {
    size_t k = random.index(count);
    T di = ic * Distance::distance(p, nodes[k], f),
      dj = jc * Distance::distance(q, nodes[k], f);
    T norm = cosine ? get_norm(nodes[k]->v, f) : 1.0;
    if (!(norm > T(0))) {
      continue;
    }
    if (di < dj) {
      aXplusbY(p->v, nodes[k]->v, p->v, (float)ic/(ic+1), (float)1/norm/(ic+1), f);
      Distance::init_node(p, f);
      ic++;
    } else if (dj < di) {
      aXplusbY(q->v, nodes[k]->v, q->v, (float)jc/(jc+1), (float)1/norm/(jc+1), f);
      Distance::init_node(q, f);
      jc++;
    }
  }
}
} // namespace

struct Base {
  template<typename T, typename S, typename Node>
  static inline void preprocess(void* nodes, size_t _s, const S node_count, const int f) {
    // Override this in specific metric structs below if you need to do any pre-processing
    // on the entire set of nodes passed into this index.
  }

  template<typename Node>
  static inline void zero_value(Node* dest) {
    // Initialize any fields that require sane defaults within this node.
  }

  template<typename T, typename Node>
  static inline void copy_node(Node* dest, const Node* source, const int f) {
    memcpy(dest->v, source->v, f * sizeof(T));
  }

  template<typename T, typename Node>
  static inline void normalize(Node* node, int f) {
    T norm = get_norm(node->v, f);
    if (norm > 0) {
      for (int z = 0; z < f; z++)
        node->v[z] /= norm;
    }
  }
};

struct Angular : Base {
  template<typename S, typename T>
  struct ANNOY_NODE_ATTRIBUTE Node {
    /*
     * We store a binary tree where each node has two things
     * - A vector associated with it
     * - Two children
     * All nodes occupy the same amount of memory
     * All nodes with n_descendants == 1 are leaf nodes.
     * A memory optimization is that for nodes with 2 <= n_descendants <= K,
     * we skip the vector. Instead we store a list of all descendants. K is
     * determined by the number of items that fits in the space of the vector.
     * For nodes with n_descendants == 1 the vector is a data point.
     * For nodes with n_descendants > K the vector is the normal of the split plane.
     * Note that we can't really do sizeof(node<T>) because we cheat and allocate
     * more memory to be able to fit the vector outside
     */
    S n_descendants;
    union {
      S children[2]; // Will possibly store more than 2
      T norm;
    };
    T v[1]; // We let this one overflow intentionally. Need to allocate at least 1 to make GCC happy
  };
  template<typename S, typename T>
  static inline T distance(const Node<S, T>* x, const Node<S, T>* y, int f) {
    // want to calculate (a/|a| - b/|b|)^2
    // = a^2 / a^2 + b^2 / b^2 - 2ab/|a||b|
    // = 2 - 2cos
    T pp = x->norm ? x->norm : dot(x->v, x->v, f); // For backwards compatibility reasons, we need to fall back and compute the norm here
    T qq = y->norm ? y->norm : dot(y->v, y->v, f);
    T pq = dot(x->v, y->v, f);
    T ppqq = pp * qq;
    if (ppqq > 0) return 2.0 - 2.0 * pq / sqrt(ppqq);
    else return 2.0; // cos is 0
  }
  template<typename S, typename T>
  static inline T margin(const Node<S, T>* n, const T* y, int f) {
    return dot(n->v, y, f);
  }
  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S, T>* n, const T* y, int f, Random& random) {
    T dot = margin(n, y, f);
    if (dot != 0)
      return (dot > 0);
    else
      return random.flip();
  }
  template<typename S, typename T, typename Random>
  static inline void create_split(const vector<Node<S, T>*>& nodes, int f, size_t s, Random& random, Node<S, T>* n) {
    Node<S, T>* p = (Node<S, T>*)malloc(s); // TODO: avoid
    Node<S, T>* q = (Node<S, T>*)malloc(s); // TODO: avoid
    two_means<T, Random, Angular, Node<S, T> >(nodes, f, random, true, p, q);
    substract(p->v, q->v, n->v, f);
    Base::normalize<T, Node<S, T> >(n, f);
    free(p);
    free(q);
  }
  template<typename T>
  static inline T normalized_distance(T distance) {
    // Used when requesting distances from Python layer
    // Turns out sometimes the squared distance is -0.0
    // so we have to make sure it's a positive number.
    return sqrt(std::max(distance, T(0)));
  }
  template<typename T>
  static inline T pq_distance(T distance, T margin, int child_nr) {
    if (child_nr == 0)
      margin = -margin;
    return std::min(distance, margin);
  }
  template<typename T>
  static inline T pq_initial_value() {
    return numeric_limits<T>::infinity();
  }
  template<typename S, typename T>
  static inline void init_node(Node<S, T>* n, int f) {
    n->norm = dot(n->v, n->v, f);
  }
  static const char* name() {
    return "angular";
  }
};


struct DotProduct : Angular {
  template<typename S, typename T>
  struct ANNOY_NODE_ATTRIBUTE Node {
    /*
     * This is an extension of the Angular node with an extra attribute for the scaled norm.
     */
    S n_descendants;
    S children[2]; // Will possibly store more than 2
    T dot_factor;
    T v[1]; // We let this one overflow intentionally. Need to allocate at least 1 to make GCC happy
  };

  static const char* name() {
    return "dot";
  }
  template<typename S, typename T>
  static inline T distance(const Node<S, T>* x, const Node<S, T>* y, int f) {
    return -dot(x->v, y->v, f);
  }

  template<typename Node>
  static inline void zero_value(Node* dest) {
    dest->dot_factor = 0;
  }

  template<typename S, typename T>
  static inline void init_node(Node<S, T>* n, int f) {
  }

  template<typename T, typename Node>
  static inline void copy_node(Node* dest, const Node* source, const int f) {
    memcpy(dest->v, source->v, f * sizeof(T));
    dest->dot_factor = source->dot_factor;
  }

  template<typename S, typename T, typename Random>
  static inline void create_split(const vector<Node<S, T>*>& nodes, int f, size_t s, Random& random, Node<S, T>* n) {
    Node<S, T>* p = (Node<S, T>*)malloc(s); // TODO: avoid
    Node<S, T>* q = (Node<S, T>*)malloc(s); // TODO: avoid
    DotProduct::zero_value(p); 
    DotProduct::zero_value(q);
    two_means<T, Random, DotProduct, Node<S, T> >(nodes, f, random, true, p, q);

    substract(p->v, q->v, n->v, f);
    n->dot_factor = p->dot_factor - q->dot_factor;

    DotProduct::normalize<T, Node<S, T> >(n, f);
    free(p);
    free(q);
  }

  template<typename T, typename Node>
  static inline void normalize(Node* node, int f) {
    T norm = sqrt(dot(node->v, node->v, f) + pow(node->dot_factor, 2));
    if (norm > 0) {
      scale(node->v, node->v, 1/norm, f);
      node->dot_factor /= norm;
    }
  }

  template<typename S, typename T>
  static inline T margin(const Node<S, T>* n, const T* y, int f) {
    return dot(n->v, y, f) + (n->dot_factor * n->dot_factor);
  }

  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S, T>* n, const T* y, int f, Random& random) {
    T dot = margin(n, y, f);
    if (dot != 0)
      return (dot > 0);
    else
      return random.flip();
  }

  template<typename T>
  static inline T normalized_distance(T distance) {
    return -distance;
  }

  template<typename T, typename S, typename Node>
  static inline void preprocess(void* nodes, size_t _s, const S node_count, const int f) {
    // This uses a method from Microsoft Research for transforming inner product spaces to cosine/angular-compatible spaces.
    // (Bachrach et al., 2014, see https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf)

    // Step one: compute the norm of each vector and store that in its extra dimension (f-1)
    for (S i = 0; i < node_count; i++) {
      Node* node = get_node_ptr<S, Node>(nodes, _s, i);
      T norm = sqrt(dot(node->v, node->v, f));
      if (isnan(norm)) norm = 0;
      node->dot_factor = norm;
    }

    // Step two: find the maximum norm
    T max_norm = 0;
    for (S i = 0; i < node_count; i++) {
      Node* node = get_node_ptr<S, Node>(nodes, _s, i);
      if (node->dot_factor > max_norm) {
        max_norm = node->dot_factor;
      }
    }

    // Step three: set each vector's extra dimension to sqrt(max_norm^2 - norm^2)
    for (S i = 0; i < node_count; i++) {
      Node* node = get_node_ptr<S, Node>(nodes, _s, i);
      T node_norm = node->dot_factor;

      T dot_factor = sqrt(pow(max_norm, static_cast<T>(2.0)) - pow(node_norm, static_cast<T>(2.0)));
      if (isnan(dot_factor)) dot_factor = 0;

      node->dot_factor = dot_factor;
    }
  }
};

struct Minkowski : Base {
  template<typename S, typename T>
  struct ANNOY_NODE_ATTRIBUTE Node {
    S n_descendants;
    T a; // need an extra constant term to determine the offset of the plane
    S children[2];
    T v[1];
  };
  template<typename S, typename T>
  static inline T margin(const Node<S, T>* n, const T* y, int f) {
    return n->a + dot(n->v, y, f);
  }
  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S, T>* n, const T* y, int f, Random& random) {
    T dot = margin(n, y, f);
    if (dot != 0)
      return (dot > 0);
    else
      return random.flip();
  }
  template<typename T>
  static inline T pq_distance(T distance, T margin, int child_nr) {
    if (child_nr == 0)
      margin = -margin;
    return std::min(distance, margin);
  }
  template<typename T>
  static inline T pq_initial_value() {
    return numeric_limits<T>::infinity();
  }
};


struct Euclidean : Minkowski {
  template<typename S, typename T>
  static inline T distance(const Node<S, T>* x, const Node<S, T>* y, int f) {
    return euclidean_distance(x->v, y->v, f);    
  }
  template<typename S, typename T, typename Random>
  static inline void create_split(const vector<Node<S, T>*>& nodes, int f, size_t s, Random& random, Node<S, T>* n) {
    Node<S, T>* p = (Node<S, T>*)malloc(s); // TODO: avoid
    Node<S, T>* q = (Node<S, T>*)malloc(s); // TODO: avoid
    two_means<T, Random, Euclidean, Node<S, T> >(nodes, f, random, false, p, q);

    for (int z = 0; z < f; z++)
      n->v[z] = p->v[z] - q->v[z];
    Base::normalize<T, Node<S, T> >(n, f);
    n->a = 0.0;
    for (int z = 0; z < f; z++)
      n->a += -n->v[z] * (p->v[z] + q->v[z]) / 2;
    free(p);
    free(q);
  }
  template<typename T>
  static inline T normalized_distance(T distance) {
    return sqrt(std::max(distance, T(0)));
  }
  template<typename S, typename T>
  static inline void init_node(Node<S, T>* n, int f) {
  }
  static const char* name() {
    return "euclidean";
  }

};


template<typename S, typename T>
class AnnoyIndexInterface {
 public:
  virtual ~AnnoyIndexInterface() {};
  virtual bool add_item(S item, const T* w, char** error=NULL) = 0;
  virtual bool build(int q, char** error=NULL) = 0;
  virtual bool unbuild(char** error=NULL) = 0;
  virtual bool save(const char* filename, bool prefault=false, char** error=NULL) = 0;
  virtual void unload() = 0;
  virtual bool load(const char* filename, bool prefault=false, char** error=NULL) = 0;
  virtual T get_distance(S i, S j) const = 0;
  virtual void get_nns_by_item(S item, size_t n, size_t search_k, vector<S>* result, vector<T>* distances) const = 0;
  virtual void get_nns_by_vector(const T* w, size_t n, size_t search_k, vector<S>* result, vector<T>* distances) const = 0;
  virtual S get_n_items() const = 0;
  virtual S get_n_trees() const = 0;
  virtual void verbose(bool v) = 0;
  virtual void get_item(S item, T* v) const = 0;
  virtual void set_seed(int q) = 0;
  virtual bool on_disk_build(const char* filename, char** error=NULL) = 0;
};

template<typename S, typename T, typename Distance, typename Random>
  class AnnoyIndex : public AnnoyIndexInterface<S, T> {
  /*
   * We use random projection to build a forest of binary trees of all items.
   * Basically just split the hyperspace into two sides by a hyperplane,
   * then recursively split each of those subtrees etc.
   * We create a tree like this q times. The default q is determined automatically
   * in such a way that we at most use 2x as much memory as the vectors take.
   */
public:
  typedef Distance D;
  typedef typename D::template Node<S, T> Node;

protected:
  const int _f;
  size_t _s;
  S _n_items;
  Random _random;
  void* _nodes; // Could either be mmapped, or point to a memory buffer that we reallocate
  S _n_nodes;
  S _nodes_size;
  vector<S> _roots;
  S _K;
  bool _loaded;
  bool _verbose;
  int _fd;
  bool _on_disk;
  bool _built;
public:

   AnnoyIndex(int f) : _f(f), _random() {
    _s = offsetof(Node, v) + _f * sizeof(T); // Size of each node
    _verbose = false;
    _built = false;
    _K = (S) (((size_t) (_s - offsetof(Node, children))) / sizeof(S)); // Max number of descendants to fit into node
    reinitialize(); // Reset everything
  }
  ~AnnoyIndex() {
    unload();
  }

  int get_f() const {
    return _f;
  }

  bool add_item(S item, const T* w, char** error=NULL) {
    return add_item_impl(item, w, error);
  }

  template<typename W>
  bool add_item_impl(S item, const W& w, char** error=NULL) {
    if (_loaded) {
      showUpdate("You can't add an item to a loaded index\n");
      if (error) *error = (char *)"You can't add an item to a loaded index";
      return false;
    }
    _allocate_size(item + 1);
    Node* n = _get(item);

    D::zero_value(n);

    n->children[0] = 0;
    n->children[1] = 0;
    n->n_descendants = 1;

    for (int z = 0; z < _f; z++)
      n->v[z] = w[z];

    D::init_node(n, _f);

    if (item >= _n_items)
      _n_items = item + 1;

    return true;
  }
    
  bool on_disk_build(const char* file, char** error=NULL) {
    _on_disk = true;
    _fd = open(file, O_RDWR | O_CREAT | O_TRUNC, (int) 0600);
    if (_fd == -1) {
      showUpdate("Error: file descriptor is -1\n");
      if (error) *error = strerror(errno);
      _fd = 0;
      return false;
    }
    _nodes_size = 1;
    if (ftruncate(_fd, _s * _nodes_size) == -1) {
      showUpdate("Error truncating file: %s\n", strerror(errno));
      if (error) *error = strerror(errno);
      return false;
    }
#ifdef MAP_POPULATE
    _nodes = (Node*) mmap(0, _s * _nodes_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, _fd, 0);
#else
    _nodes = (Node*) mmap(0, _s * _nodes_size, PROT_READ | PROT_WRITE, MAP_SHARED, _fd, 0);
#endif
    return true;
  }
    
  bool build(int q, char** error=NULL) {
    if (_loaded) {
      showUpdate("You can't build a loaded index\n");
      if (error) *error = (char *)"You can't build a loaded index";
      return false;
    }

    if (_built) {
      showUpdate("You can't build a built index\n");
      if (error) *error = (char *)"You can't build a built index";
      return false;
    }

    D::template preprocess<T, S, Node>(_nodes, _s, _n_items, _f);

    _n_nodes = _n_items;
    while (1) {
      if (q == -1 && _n_nodes >= _n_items * 2)
        break;
      if (q != -1 && _roots.size() >= (size_t)q)
        break;
      if (_verbose) showUpdate("pass %zd...\n", _roots.size());

      vector<S> indices;
      for (S i = 0; i < _n_items; i++) {
        if (_get(i)->n_descendants >= 1) // Issue #223
          indices.push_back(i);
      }

      _roots.push_back(_make_tree(indices, true));
    }

    // Also, copy the roots into the last segment of the array
    // This way we can load them faster without reading the whole file
    _allocate_size(_n_nodes + (S)_roots.size());
    for (size_t i = 0; i < _roots.size(); i++)
      memcpy(_get(_n_nodes + (S)i), _get(_roots[i]), _s);
    _n_nodes += _roots.size();

    if (_verbose) showUpdate("has %d nodes\n", _n_nodes);
    
    if (_on_disk) {
      _nodes = remap_memory(_nodes, _fd, _s * _nodes_size, _s * _n_nodes);
      if (ftruncate(_fd, _s * _n_nodes)) {
	// TODO: this probably creates an index in a corrupt state... not sure what to do
	showUpdate("Error truncating file: %s\n", strerror(errno));
	if (error) *error = strerror(errno);
	return false;
      }
      _nodes_size = _n_nodes;
    }
    _built = true;
    return true;
  }
  
  bool unbuild(char** error=NULL) {
    if (_loaded) {
      showUpdate("You can't unbuild a loaded index\n");
      if (error) *error = (char *)"You can't unbuild a loaded index";
      return false;
    }

    _roots.clear();
    _n_nodes = _n_items;
    _built = false;

    return true;
  }

  bool save(const char* filename, bool prefault=false, char** error=NULL) {
    if (!_built) {
      showUpdate("You can't save an index that hasn't been built\n");
      if (error) *error = (char *)"You can't save an index that hasn't been built";
      return false;
    }
    if (_on_disk) {
      return true;
    } else {
      // Delete file if it already exists (See issue #335)
      unlink(filename);

      printf("path: %s\n", filename);

      FILE *f = fopen(filename, "wb");
      if (f == NULL) {
        showUpdate("Unable to open: %s\n", strerror(errno));
        if (error) *error = strerror(errno);
        return false;
      }

      if (fwrite(_nodes, _s, _n_nodes, f) != (size_t) _n_nodes) {
        showUpdate("Unable to write: %s\n", strerror(errno));
        if (error) *error = strerror(errno);
        return false;
      }

      if (fclose(f) == EOF) {
        showUpdate("Unable to close: %s\n", strerror(errno));
        if (error) *error = strerror(errno);
        return false;
      }

      unload();
      return load(filename, prefault, error);
    }
  }

  void reinitialize() {
    _fd = 0;
    _nodes = NULL;
    _loaded = false;
    _n_items = 0;
    _n_nodes = 0;
    _nodes_size = 0;
    _on_disk = false;
    _roots.clear();
  }

  void unload() {
    if (_on_disk && _fd) {
      close(_fd);
      munmap(_nodes, _s * _nodes_size);
    } else {
      if (_fd) {
        // we have mmapped data
        close(_fd);
        munmap(_nodes, _n_nodes * _s);
      } else if (_nodes) {
        // We have heap allocated data
        free(_nodes);
      }
    }
    reinitialize();
    if (_verbose) showUpdate("unloaded\n");
  }

  bool load(const char* filename, bool prefault=false, char** error=NULL) {
    _fd = open(filename, O_RDONLY, (int)0400);
    if (_fd == -1) {
      showUpdate("Error: file descriptor is -1\n");
      if (error) *error = strerror(errno);
      _fd = 0;
      return false;
    }
    off_t size = lseek(_fd, 0, SEEK_END);
    if (size == -1) {
      showUpdate("lseek returned -1\n");
      if (error) *error = strerror(errno);
      return false;
    } else if (size == 0) {
      showUpdate("Size of file is zero\n");
      if (error) *error = (char *)"Size of file is zero";
      return false;
    } else if (size % _s) {
      // Something is fishy with this index!
      showUpdate("Error: index size %zu is not a multiple of vector size %zu\n", (size_t)size, _s);
      if (error) *error = (char *)"Index size is not a multiple of vector size";
      return false;
    }

    int flags = MAP_SHARED;
    if (prefault) {
#ifdef MAP_POPULATE
      flags |= MAP_POPULATE;
#else
      showUpdate("prefault is set to true, but MAP_POPULATE is not defined on this platform");
#endif
    }
    _nodes = (Node*)mmap(0, size, PROT_READ, flags, _fd, 0);
    _n_nodes = (S)(size / _s);

    // Find the roots by scanning the end of the file and taking the nodes with most descendants
    _roots.clear();
    S m = -1;
    for (S i = _n_nodes - 1; i >= 0; i--) {
      S k = _get(i)->n_descendants;
      if (m == -1 || k == m) {
        _roots.push_back(i);
        m = k;
      } else {
        break;
      }
    }
    // hacky fix: since the last root precedes the copy of all roots, delete it
    if (_roots.size() > 1 && _get(_roots.front())->children[0] == _get(_roots.back())->children[0])
      _roots.pop_back();
    _loaded = true;
    _built = true;
    _n_items = m;
    if (_verbose) showUpdate("found %lu roots with degree %d\n", _roots.size(), m);
    return true;
  }

  T get_distance(S i, S j) const {
    return D::normalized_distance(D::distance(_get(i), _get(j), _f));
  }

  void get_nns_by_item(S item, size_t n, size_t search_k, vector<S>* result, vector<T>* distances) const {
    // TODO: handle OOB
    const Node* m = _get(item);
    _get_all_nns(m->v, n, search_k, result, distances);
  }

  void get_nns_by_vector(const T* w, size_t n, size_t search_k, vector<S>* result, vector<T>* distances) const {
    _get_all_nns(w, n, search_k, result, distances);
  }

  S get_n_items() const {
    return _n_items;
  }

  S get_n_trees() const {
    return _roots.size();
  }

  void verbose(bool v) {
    _verbose = v;
  }

  void get_item(S item, T* v) const {
    // TODO: handle OOB
    Node* m = _get(item);
    memcpy(v, m->v, (_f) * sizeof(T));
  }

  void set_seed(int seed) {
    _random.set_seed(seed);
  }

protected:
  void _allocate_size(S n) {
    if (n > _nodes_size) {
      const double reallocation_factor = 1.3;
      S new_nodes_size = std::max(n, (S) ((_nodes_size + 1) * reallocation_factor));
      void *old = _nodes;
      
      if (_on_disk) {
        int rc = ftruncate(_fd, _s * new_nodes_size);
        if (_verbose && rc) showUpdate("File truncation error\n");
        _nodes = remap_memory(_nodes, _fd, _s * _nodes_size, _s * new_nodes_size);
      } else {
        _nodes = realloc(_nodes, _s * new_nodes_size);
        memset((char *) _nodes + (_nodes_size * _s) / sizeof(char), 0, (new_nodes_size - _nodes_size) * _s);
      }
      
      _nodes_size = new_nodes_size;
      if (_verbose) showUpdate("Reallocating to %d nodes: old_address=%p, new_address=%p\n", new_nodes_size, old, _nodes);
    }
  }

  inline Node* _get(const S i) const {
    return get_node_ptr<S, Node>(_nodes, _s, i);
  }

  /* ===============================================================================================================
   * Indexing related functions
   * ===============================================================================================================*/
  inline void _simd_matrix_dot_kernel2(Node* v_node, vector<S> indices, vector<S>* children_indices, int f, int start_idx) const {
      float *x1 = _get(indices[start_idx])->v;
      float *x2 = _get(indices[start_idx+1])->v;
      float *x3 = _get(indices[start_idx+2])->v;
      float *x4 = _get(indices[start_idx+3])->v;
      float *x5 = _get(indices[start_idx+4])->v;

      float *n = v_node->v;

      __m256 t1 = _mm256_setzero_ps();
      __m256 t2 = _mm256_setzero_ps();
      __m256 t3 = _mm256_setzero_ps();
      __m256 t4 = _mm256_setzero_ps();
      __m256 t5 = _mm256_setzero_ps();

      float o1=0, o2=0, o3=0, o4=0, o5=0;

      if (f>7) {
          for (; f > 7; f -= 8) {
              __m256 n_chunk = _mm256_loadu_ps(n);
              __m256 v1 = _mm256_loadu_ps(x1);
              __m256 v2 = _mm256_loadu_ps(x2);
              __m256 v3 = _mm256_loadu_ps(x3);
              __m256 v4 = _mm256_loadu_ps(x4);
              __m256 v5 = _mm256_loadu_ps(x5);

              t1 = _mm256_fmadd_ps(v1, n_chunk, t1);
              t2 = _mm256_fmadd_ps(v2, n_chunk, t2);
              t3 = _mm256_fmadd_ps(v3, n_chunk, t3);
              t4 = _mm256_fmadd_ps(v4, n_chunk, t4);
              t5 = _mm256_fmadd_ps(v5, n_chunk, t5);

              x1 += 8;
              x2 += 8;
              x3 += 8;
              x4 += 8;
              x5 += 8;
              n += 8;
          }

          o1 = hsum256_ps_avx(t1);
          o2 = hsum256_ps_avx(t2);
          o3 = hsum256_ps_avx(t3);
          o4 = hsum256_ps_avx(t4);
          o5 = hsum256_ps_avx(t5);
      }
      for (; f > 0; f--) {
          o1 += *x1 * *n;
          o2 += *x2 * *n;
          o3 += *x3 * *n;
          o4 += *x4 * *n;
          o5 += *x5 * *n;

          x1 ++;
          x2 ++;
          x3 ++;
          x4 ++;
          x5 ++;
          n++;
      }

      float n_term = v_node->dot_factor * v_node->dot_factor;
      o1 += n_term;
      o2 += n_term;
      o3 += n_term;
      o4 += n_term;
      o5 += n_term;

      children_indices[o1>0].push_back(indices[start_idx]);
      children_indices[o2>0].push_back(indices[start_idx+1]);
      children_indices[o3>0].push_back(indices[start_idx+2]);
      children_indices[o4>0].push_back(indices[start_idx+3]);
      children_indices[o5>0].push_back(indices[start_idx+4]);
  }

  inline void _partition_points(Node* v_node, vector<S> indices, vector<S>* children_indices) {
      int nv = indices.size();
      int i = 0;

//      if (nv <= 200){
//          for (; nv > 4; nv -= 5) {
//              _simd_matrix_dot_kernel2(v_node, indices, children_indices, _f, i);
//              i += 5;
//          }
//      }

      float n_term = v_node->dot_factor * v_node->dot_factor;
      for (; nv > 0; nv--) {
          S j = indices[i];
          bool side = (dot(v_node->v, _get(j)->v, _f) + n_term) > 0;
          children_indices[side].push_back(j);
          i++;
      }
  }


  S _make_tree(const vector<S >& indices, bool is_root) {
    // The basic rule is that if we have <= _K items, then it's a leaf node, otherwise it's a split node.
    // There's some regrettable complications caused by the problem that root nodes have to be "special":
    // 1. We identify root nodes by the arguable logic that _n_items == n->n_descendants, regardless of how many descendants they actually have
    // 2. Root nodes with only 1 child need to be a "dummy" parent
    // 3. Due to the _n_items "hack", we need to be careful with the cases where _n_items <= _K or _n_items > _K
    if (indices.size() == 1 && !is_root)
      return indices[0];

    if (indices.size() <= (size_t)_K && (!is_root || (size_t)_n_items <= (size_t)_K || indices.size() == 1)) {
      _allocate_size(_n_nodes + 1);
      S item = _n_nodes++;
      Node* m = _get(item);
      m->n_descendants = is_root ? _n_items : (S)indices.size();

      // Using std::copy instead of a loop seems to resolve issues #3 and #13,
      // probably because gcc 4.8 goes overboard with optimizations.
      // Using memcpy instead of std::copy for MSVC compatibility. #235
      // Only copy when necessary to avoid crash in MSVC 9. #293
      if (!indices.empty())
        memcpy(m->children, &indices[0], indices.size() * sizeof(S));
      return item;
    }

    vector<Node*> children;
    for (size_t i = 0; i < indices.size(); i++) {
      S j = indices[i];
      Node* n = _get(j);
      if (n)
        children.push_back(n);
    }

    vector<S> children_indices[2];
    Node* m = (Node*)malloc(_s); // TODO: avoid

    D::create_split(children, _f, _s, _random, m);

    // TODO. change this part.
    _partition_points(m, indices, children_indices);

    // If we didn't find a hyperplane, just randomize sides as a last option
    while (children_indices[0].size() == 0 || children_indices[1].size() == 0) {
      if (_verbose)
        showUpdate("\tNo hyperplane found (left has %ld children, right has %ld children)\n",
          children_indices[0].size(), children_indices[1].size());
      if (_verbose && indices.size() > 100000)
        showUpdate("Failed splitting %lu items\n", indices.size());

      children_indices[0].clear();
      children_indices[1].clear();

      // Set the vector to 0.0
      for (int z = 0; z < _f; z++)
        m->v[z] = 0.0;

      for (size_t i = 0; i < indices.size(); i++) {
        S j = indices[i];
        // Just randomize...
        children_indices[_random.flip()].push_back(j);
      }
    }

    int flip = (children_indices[0].size() > children_indices[1].size());

    m->n_descendants = is_root ? _n_items : (S)indices.size();
    for (int side = 0; side < 2; side++) {
      // run _make_tree for the smallest child first (for cache locality)
      m->children[side^flip] = _make_tree(children_indices[side^flip], false);
    }

    _allocate_size(_n_nodes + 1);
    S item = _n_nodes++;
    memcpy(_get(item), m, _s);
    free(m);

    return item;
  }

  /* ===============================================================================================================
   * Search related functions
   * ===============================================================================================================*/

//  inline void _get_distances(Node* v_node, vector<S> nns, vector<pair<T, S> >* nns_dist) const {
//      for (size_t i = 0; i < nns.size(); i++) {
//          S j = nns[i];
//          nns_dist->push_back(make_pair(D::distance(v_node, _get(j), _f), j));
//      }
//  }

  inline void _simd_matrix_dot_kernel(Node* v_node, vector<S> indices,  vector<pair<T, S> >* nns_dist, int f, int start_idx) const {
      float *x1 = _get(indices[start_idx])->v;
      float *x2 = _get(indices[start_idx+1])->v;
      float *x3 = _get(indices[start_idx+2])->v;
      float *x4 = _get(indices[start_idx+3])->v;
      float *x5 = _get(indices[start_idx+4])->v;
      float *x6 = _get(indices[start_idx+5])->v;
      float *x7 = _get(indices[start_idx+6])->v;
      float *x8 = _get(indices[start_idx+7])->v;
      float *x9 = _get(indices[start_idx+8])->v;
      float *x10 = _get(indices[start_idx+9])->v;

      float *n = v_node->v;

      __m256 t1 = _mm256_setzero_ps();
      __m256 t2 = _mm256_setzero_ps();
      __m256 t3 = _mm256_setzero_ps();
      __m256 t4 = _mm256_setzero_ps();
      __m256 t5 = _mm256_setzero_ps();
      __m256 t6 = _mm256_setzero_ps();
      __m256 t7 = _mm256_setzero_ps();
      __m256 t8 = _mm256_setzero_ps();
      __m256 t9 = _mm256_setzero_ps();
      __m256 t10 = _mm256_setzero_ps();

      float o1=0, o2=0, o3=0, o4=0, o5=0, o6=0, o7=0, o8=0, o9=0, o10=0;

      if (f>7) {
          for (; f > 7; f -= 8) {
              __m256 n_chunk = _mm256_loadu_ps(n);
              __m256 v1 = _mm256_loadu_ps(x1);
              __m256 v2 = _mm256_loadu_ps(x2);
              __m256 v3 = _mm256_loadu_ps(x3);
              __m256 v4 = _mm256_loadu_ps(x4);
              __m256 v5 = _mm256_loadu_ps(x5);

              t1 = _mm256_fmadd_ps(v1, n_chunk, t1);
              t2 = _mm256_fmadd_ps(v2, n_chunk, t2);
              t3 = _mm256_fmadd_ps(v3, n_chunk, t3);
              t4 = _mm256_fmadd_ps(v4, n_chunk, t4);
              t5 = _mm256_fmadd_ps(v5, n_chunk, t5);
              t6 = _mm256_fmadd_ps(_mm256_loadu_ps(x6), n_chunk, t6);
              t7 = _mm256_fmadd_ps(_mm256_loadu_ps(x7), n_chunk, t7);
              t8 = _mm256_fmadd_ps(_mm256_loadu_ps(x8), n_chunk, t8);
              t9 = _mm256_fmadd_ps(_mm256_loadu_ps(x9), n_chunk, t9);
              t10 = _mm256_fmadd_ps(_mm256_loadu_ps(x10), n_chunk, t10);

              x1 += 8;
              x2 += 8;
              x3 += 8;
              x4 += 8;
              x5 += 8;
              x6 += 8;
              x7 += 8;
              x8 += 8;
              x9 += 8;
              x10 += 8;
              n += 8;
          }

          o1 = hsum256_ps_avx(t1);
          o2 = hsum256_ps_avx(t2);
          o3 = hsum256_ps_avx(t3);
          o4 = hsum256_ps_avx(t4);
          o5 = hsum256_ps_avx(t5);
          o6 = hsum256_ps_avx(t6);
          o7 = hsum256_ps_avx(t7);
          o8 = hsum256_ps_avx(t8);
          o9 = hsum256_ps_avx(t9);
          o10 = hsum256_ps_avx(t10);
      }
      for (; f > 0; f--) {
          o1 += *x1 * *n;
          o2 += *x2 * *n;
          o3 += *x3 * *n;
          o4 += *x4 * *n;
          o5 += *x5 * *n;
          o6 += *x6 * *n;
          o7 += *x7 * *n;
          o8 += *x8 * *n;
          o9 += *x9 * *n;
          o10 += *x10 * *n;

          x1 ++;
          x2 ++;
          x3 ++;
          x4 ++;
          x5 ++;
          x6 ++;
          x7 ++;
          x8 ++;
          x9 ++;
          x10 ++;
          n++;
      }
      nns_dist->push_back(make_pair(-o1, indices[start_idx]));
      nns_dist->push_back(make_pair(-o2, indices[start_idx+1]));
      nns_dist->push_back(make_pair(-o3, indices[start_idx+2]));
      nns_dist->push_back(make_pair(-o4, indices[start_idx+3]));
      nns_dist->push_back(make_pair(-o5, indices[start_idx+4]));
      nns_dist->push_back(make_pair(-o6, indices[start_idx+5]));
      nns_dist->push_back(make_pair(-o7, indices[start_idx+6]));
      nns_dist->push_back(make_pair(-o8, indices[start_idx+7]));
      nns_dist->push_back(make_pair(-o9, indices[start_idx+8]));
      nns_dist->push_back(make_pair(-o10, indices[start_idx+9]));
  }

  inline void _get_distances(Node* v_node, vector<S> nns, vector<pair<T, S> >* nns_dist) const {
      int nv = nns.size();
      int i = 0;
      for (; nv > 9; nv -= 10) {
          _simd_matrix_dot_kernel(v_node, nns, nns_dist, _f, i);
          i += 10;
      }
      for (; nv > 0; nv--) {
          S j = nns[i];
          nns_dist->push_back(make_pair(-dot(v_node->v, _get(j)->v, _f), j));
          i++;
      }
  }

  void _get_all_nns(const T* v, size_t n, size_t search_k, vector<S>* result, vector<T>* distances) const {
    Node* v_node = (Node *)malloc(_s); // TODO: avoid
    D::template zero_value<Node>(v_node);
    memcpy(v_node->v, v, sizeof(T) * _f);
    D::init_node(v_node, _f);

    std::priority_queue<pair<T, S> > q;

    if (search_k == (size_t)-1) {
      search_k = n * _roots.size();
    }

    for (size_t i = 0; i < _roots.size(); i++) {
      q.push(make_pair(Distance::template pq_initial_value<T>(), _roots[i]));
    }

    std::vector<S> nns;
    while (nns.size() < search_k && !q.empty()) {
      const pair<T, S>& top = q.top();
      T d = top.first;
      S i = top.second;
      Node* nd = _get(i);
      q.pop();
      if (nd->n_descendants == 1 && i < _n_items) {
        nns.push_back(i);
      } else if (nd->n_descendants <= _K) {
        const S* dst = nd->children;
        nns.insert(nns.end(), dst, &dst[nd->n_descendants]);
      } else {
        T margin = D::margin(nd, v, _f);
        q.push(make_pair(D::pq_distance(d, margin, 1), static_cast<S>(nd->children[1])));
        q.push(make_pair(D::pq_distance(d, margin, 0), static_cast<S>(nd->children[0])));
      }
    }

    // Get distances for all items
    // To avoid calculating distance multiple times for any items, sort by id
    // TODO. make this part better
    std::sort(nns.begin(), nns.end());
    nns.erase( unique( nns.begin(), nns.end() ), nns.end() );

    vector<pair<T, S> > nns_dist;
    _get_distances(v_node, nns, &nns_dist);

    size_t m = nns_dist.size();
    size_t p = n < m ? n : m; // Return this many items
    std::partial_sort(nns_dist.begin(), nns_dist.begin() + p, nns_dist.end());
    for (size_t i = 0; i < p; i++) {
      if (distances)
        distances->push_back(D::normalized_distance(nns_dist[i].first));
      result->push_back(nns_dist[i].second);
    }
    free(v_node);
  }
};

#endif
// vim: tabstop=2 shiftwidth=2
