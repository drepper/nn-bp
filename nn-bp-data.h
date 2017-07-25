#ifndef _NN_BP_DATA_H
#define _NN_BP_DATA_H 1


template<typename _NN>
struct train_data {
  template<typename... _Gen>
  train_data(std::size_t n_, typename _NN::an_t (*f)(std::size_t), _Gen... generators) : n(n_), nclasses(sizeof...(generators)) {
    typename _NN::an_t tmp;
    if (int(sizeof...(generators)) > tmp.rows())
      throw std::runtime_error("number of generators greater than number of output neurons");
    init(sizeof...(generators), n);
    generate(0, n, f, generators...);
  }

  _NN& train(_NN& nn, std::size_t nepochs = 1) const {
    nn.train(nepochs, input.begin(), input.end(), expected.begin());
    return nn;
  }

  template<unsigned N>
  auto cbegin() const { return input.cbegin() + n * N; }
  template<unsigned N>
  auto cend() const { return input.cbegin() + n * (N + 1); }

  const size_t nclasses;

private:
  void init(std::size_t nclasses, std::size_t n) {
    input.resize(nclasses * n);
    expected.resize(nclasses * n);
  }

  template<typename _Gen1, typename... _Gen2>
  void generate(std::size_t classnr, std::size_t n, typename _NN::an_t (*f)(std::size_t), _Gen1 gen1, _Gen2... generators) {
    std::generate_n(input.begin() + n * classnr, n, gen1);
    std::fill_n(expected.begin() + n * classnr, n, f(classnr));
    generate(classnr + 1, n, f, generators...);
  }

  void generate(std::size_t, std::size_t, typename _NN::an_t (*)(std::size_t)) { }

  const std::size_t n;
  std::vector<typename _NN::template a_t<0>> input;
  std::vector<typename _NN::an_t> expected;
};


template<typename _NN>
struct repeated_train_data {
  std::size_t n;
  const train_data<_NN>& td;
};


template<typename _Float, int _N, int _M, int... _Sizes>
Network<_Float,_N,_M,_Sizes...>& operator<<(Network<_Float,_N,_M,_Sizes...>& nn, const train_data<Network<_Float,_N,_M,_Sizes...>>& td) {
  return td.train(nn);
}


template<class _NN>
repeated_train_data<_NN> operator*(std::size_t n, const train_data<_NN>& td) {
  return repeated_train_data<_NN> { n, td };
}


template<class _NN>
repeated_train_data<_NN> operator*(const train_data<_NN>& td, std::size_t n) {
  return repeated_train_data<_NN> { n, td };
}


template<typename _Float, int _N, int _M, int... _Sizes>
Network<_Float,_N,_M,_Sizes...>& operator<<(Network<_Float,_N,_M,_Sizes...>& nn, const repeated_train_data<Network<_Float,_N,_M,_Sizes...>>& rtd) {
  return rtd.td.train(nn, rtd.n);
}

#endif // nn-bp-data.h
// Local Variables:
//  mode: c++
// End:
