#ifndef _NN_BP_H
#define _NN_BP_H 1

#include <algorithm>
#include <type_traits>
#include <ext/random>
#include <Eigen/Eigen>


template<typename _Float, int _N>
  Eigen::Matrix<typename std::enable_if<std::is_floating_point<_Float>::value, _Float>::type,(_N>4)?Eigen::Dynamic:_N,1>
  sigmoid(const Eigen::Matrix<_Float,(_N>4)?Eigen::Dynamic:_N,1>& z) {
    return _Float(1) / ((-z).array().exp() + _Float(1));
  }


template<typename _Float, int _N, int _Nd = (_N>4)?Eigen::Dynamic:_N>
  Eigen::Matrix<typename std::enable_if<std::is_floating_point<_Float>::value, _Float>::type,_Nd,1>
  sigmoid_diff(const Eigen::Matrix<_Float,_Nd,1>& z) {
    const Eigen::Array<_Float,_Nd,1> s = sigmoid<_Float,_N>(z);
    return (s * (_Float(1) - s)).matrix();
  }


// The selected cost function is 1/2 || a_n - y(x) ||^2
//
// This means the derivative is a_n - y(x)
template<typename _Float, int _N, int _Nd = (_N>4)?Eigen::Dynamic:_N>
  Eigen::Array<typename std::enable_if<std::is_floating_point<_Float>::value, _Float>::type,_Nd,1>
    cost_diff(const Eigen::Array<_Float,_Nd,1>& an, const Eigen::Array<_Float,_Nd,1>& y) {
    return an - y;
  }


template<typename _Float, int _N, int _M, int... _Sizes>
class Network {
public:
  typedef _Float float_type;
  static_assert(std::is_floating_point<float_type>::value, "_Float parameter must be floating-point type");
  static_assert(_N > 0, "First layer size must be positive");
  static_assert(_M > 0, "Second layer size must be positive");

private:
  template<int _Idx, int... _LSizes>
  struct Layer_Type;

  template<int... LSizes>
  struct Layer;


  template<int _L1, int... _LSizes>
  struct Layer_Type<0,_L1,_LSizes...> {
    static_assert(_L1 > 0, "layer size must be positive");
    static constexpr int layer_size = _L1;
    static constexpr int _L1d = _L1 > 4 ? Eigen::Dynamic : _L1;
    typedef Eigen::Matrix<float_type,_L1d,1> an_t;

    static an_t eval(const Layer<_L1,_LSizes...>& layer, const an_t& input) {
      return input;
    }
  };

  template<int _Idx, int _L1, int... _LSizes>
  struct Layer_Type<_Idx,_L1,_LSizes...> : public Layer_Type<_Idx-1,_LSizes...> {
    static constexpr int layer_size = _L1;
    static constexpr int _L1d = _L1 > 4 ? Eigen::Dynamic : _L1;
    typedef Layer_Type<_Idx-1,_LSizes...> base_class;
    typedef Eigen::Matrix<float_type,_L1d,1> a_t;
    typedef typename base_class::an_t an_t;

    static an_t eval(const Layer<_L1,_LSizes...>& layer, const a_t& input) {
      return base_class::eval(layer, sigmoid<float_type,base_class::layer_size>(layer.w * input + layer.b));
    }
  };


  template<int _N1>
  class Layer<_N1> {
    static_assert(_N1 != 0, "layer size cannot be zero");
    static constexpr int _N1d = _N1 > 4 ? Eigen::Dynamic : _N1;
  protected:
    typedef Eigen::Array<float_type,_N1d,1> an_t;

  public:
    an_t train_add1(const an_t& an, const an_t& y) const {
      return cost_diff<float_type,_N1>(an, y);
    }

    void new_batch() const { }

    void finalize_batch(double, double, bool) const { }

    std::size_t get_nnodes(std::size_t n) const {
      if (n == 0)
	return _N1;
      throw std::runtime_error("invalid index");
    }

    template<typename _CharT, typename _Traits>
    std::basic_istream<_CharT,_Traits>&
      load(std::basic_istream<_CharT,_Traits>& is) { return is; }
    template<typename _CharT, typename _Traits>
    std::basic_istream<_CharT,_Traits>&
      save(std::basic_istream<_CharT,_Traits>& os) const { return os; }
  };

  template<int _N1, int _N2, int... _LSizes>
  class Layer<_N1,_N2,_LSizes...> : public Layer<_N2,_LSizes...> {
    static_assert(_N1 != 0, "layer size cannot be zero");
    static constexpr int _N1d = _N1 > 4 ? Eigen::Dynamic : _N1;
    static constexpr int _N2d = _N2 > 4 ? Eigen::Dynamic : _N2;
    typedef Layer<_N2,_LSizes...> base_class;

    template<int _Idx, int... _LLSizes>
    friend struct Layer_Type;

    typedef Eigen::Matrix<float_type,_N2d,_N1d> w_t;
    typedef Eigen::Matrix<float_type,_N1d,1> a_t;
    typedef Eigen::Matrix<float_type,_N2d,1> y_t;
  protected:
    typedef typename base_class::an_t an_t;

  public:
    Layer();

    a_t train_add1(const a_t& a, const an_t& an) {
      y_t z = w * a + b;
      y_t y = sigmoid<float_type,_N2>(z);

      y_t delta = base_class::train_add1(y, an).array() * sigmoid_diff<float_type,_N2>(z).array();

      nabla_b += delta;
      nabla_w += delta * a.transpose();

      return w.transpose() * delta;
    }

    void clear_nabla() {
      nabla_w.fill(float_type(0));
      nabla_b.fill(float_type(0));
    }

    void new_batch() {
      clear_nabla();
      base_class::new_batch();
    }

    std::size_t get_nnodes(std::size_t n) const {
      return n == 0 ? _N1 : base_class::get_nnodes(n - 1);
    }

    void finalize_batch(float_type agefact, float_type eta, bool clearp = true) {
      base_class::finalize_batch(agefact, eta, clearp);

      w -= (eta * (nabla_w.array() + w.array() * agefact)).matrix();
      b -= (eta * nabla_b.array()).matrix();

      if (clearp)
	clear_nabla();
    }

    template<typename _CharT, typename _Traits>
    std::basic_istream<_CharT,_Traits>&
      load(std::basic_istream<_CharT,_Traits>& is) {
      is.read(w.data(), sizeof(w_t::Scalar) * w.size());
      is.read(b.data(), sizeof(y_t::Scalar) * b.size());
      return is;
    }
    template<typename _CharT, typename _Traits>
    std::basic_istream<_CharT,_Traits>&
      save(std::basic_istream<_CharT,_Traits>& os) const {
      os.write(w.data(), sizeof(w_t::Scalar) * w.size());
      os.write(b.data(), sizeof(y_t::Scalar) * b.size());
      return os;
    }

  private:
    w_t w;
    y_t b;

    w_t nabla_w;
    y_t nabla_b;
  };

  typedef Layer<_N,_M,_Sizes...> layer_t;

  static constexpr int _Nd = _N > 4 ? Eigen::Dynamic : _N;
  static constexpr int _Md = _M > 4 ? Eigen::Dynamic : _M;

public:
  static constexpr int nlayers = 1 + sizeof...(_Sizes);
  typedef Eigen::Matrix<float_type,_Md,_Nd> w_t;
  typedef Eigen::Matrix<float_type,_Nd,1> a0_t;
  template<std::size_t _L>
    using a_t = typename Layer_Type<_L,_N,_M,_Sizes...>::an_t;
  typedef typename Layer_Type<nlayers,_N,_M,_Sizes...>::an_t an_t;

  Network(size_t batchsize_, float_type eta_ = 0.1, float_type lambda_ = 0.0001) : batchsize(batchsize_), eta(eta_), lambda(lambda_) {
    if (batchsize == 0)
      throw std::runtime_error("batch size must be positive");
  }

  template<class _Iter_a0, class _Iter_an>
  void train(std::size_t nepoch, _Iter_a0 ibegin, _Iter_a0 iend, _Iter_an obegin) {
    std::size_t ntrain = std::distance(ibegin, iend);
    if (ntrain > 0) {
      std::vector<std::size_t> iidx(ntrain);
      std::iota(iidx.begin(), iidx.end(), std::size_t(0));

      __gnu_cxx::sfmt19937 e((std::random_device())());
      for (std::size_t epoch = 0; epoch < nepoch; ++epoch) {
	std::shuffle(iidx.begin(), iidx.end(), e);

	layers.new_batch();
	std::size_t inbatch = 0;
	for (auto idx : iidx) {
	  (void) layers.train_add1(ibegin[idx], obegin[idx]);

	  if (++inbatch == batchsize) {
	    layers.finalize_batch(lambda * inbatch / ntrain, eta);
	    inbatch = 0;
	  }
	}

	if (inbatch != 0)
	  layers.finalize_batch(lambda * inbatch / ntrain, eta, false);

	++nepochs;
      }
    }
  }

  std::size_t get_nepochs() const { return nepochs; }

  std::size_t get_nnodes(std::size_t n) const { return layers.get_nnodes(n); }

  template<std::size_t _LE>
    typename Layer_Type<_LE,_N,_M,_Sizes...>::an_t eval(const a0_t& input) const {
      return Layer_Type<_LE,_N,_M,_Sizes...>::eval(layers, input);
    }

  template<typename _CharT, typename _Traits>
  std::basic_istream<_CharT,_Traits>&
    load(std::basic_istream<_CharT,_Traits>& is) {
      return layers.load(is);
  }
  template<typename _CharT, typename _Traits>
  std::basic_istream<_CharT,_Traits>&
    save(std::basic_istream<_CharT,_Traits>& is) {
      return layers.save(is);
  }
private:
  layer_t layers;

  const std::size_t batchsize;
  const float_type eta;
  const float_type lambda;
  std::size_t nepochs = 0;
};


template<std::size_t _LE, typename _Float, int _N, int _M, int... _Sizes>
  auto eval(const Network<_Float,_N,_M,_Sizes...>& nn, const typename Network<_Float,_N,_M,_Sizes...>::a0_t& input) {
    return nn.template eval<_LE>(input);
  }


template<typename _Float, int _N, int _M, int... _Sizes>
  auto eval(const Network<_Float,_N,_M,_Sizes...>& nn, const typename Network<_Float,_N,_M,_Sizes...>::a0_t& input) {
    return nn.template eval<1+sizeof...(_Sizes)>(input);
  }


#include "nn-bp.tcc"

#endif // nn-bp.h
// Local Variables:
//  mode: c++
// End:
