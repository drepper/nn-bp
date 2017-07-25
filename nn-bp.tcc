template<typename _Float, int _N, int _M, int... _Sizes>
  template<int _N1, int _N2, int... _LSizes>
    Network<_Float,_N,_M,_Sizes...>::Layer<_N1,_N2,_LSizes...>::Layer() {
      if (_N1 != _N1d || _N2 != _N2d) {
	w.resize(_N2, _N1);
	nabla_w.resize(_N2, _N1);
      }
      if (_N2 != _N2d) {
	b.resize(_N2, 1);
	nabla_b.resize(_N2, 1);
      }

      __gnu_cxx::sfmt19937 e((std::random_device())());
      std::normal_distribution<float_type> d;

      std::generate(w.data(), w.data() + w.size(), [&d,&e]{return d(e);});
      std::generate(b.data(), b.data() + b.size(), [&d,&e]{return d(e);});
    }

// Local Variables:
//  mode: c++
// End:
