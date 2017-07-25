// Compile with -DEIGEN_NO_DEBUG

#include <iostream>
#include <set>
#include <algorithm>
#include <gtkmm.h>
#include "nn-bp.h"
#include "nn-bp-data.h"


namespace {

  __gnu_cxx::sfmt19937 eg((std::random_device())());
  std::normal_distribution<double> ed;

#if 0
  Eigen::Matrix<double,2,1> f1(double x) {
    return Eigen::Matrix<double,2,1>(x, x * x - 0.5 + ed(eg));
  }

  Eigen::Matrix<double,2,1> f2(double x) {
    return Eigen::Matrix<double,2,1>(x, x * x - 1.0 + ed(eg));
  }
#elif 1
  Eigen::Matrix<double,2,1> f1(double x) {
    return Eigen::Matrix<double,2,1>(x, std::fabs(std::sin(x * __gnu_cxx::__math_constants<double>::__pi)) - 0.5 + ed(eg));
  }

  Eigen::Matrix<double,2,1> f2(double x) {
    return Eigen::Matrix<double,2,1>(x, std::fabs(std::sin(x * __gnu_cxx::__math_constants<double>::__pi)) - 1.0 + ed(eg));
  }
#else
  Eigen::Matrix<double,2,1> f1(double x) {
    return Eigen::Matrix<double,2,1>(x, std::fabs(std::sin((x + 1.0) * 2.0 / 3.0 * __gnu_cxx::__math_constants<double>::__pi)) - 0.5 + ed(eg));
  }

  Eigen::Matrix<double,2,1> f2(double x) {
    return Eigen::Matrix<double,2,1>(x, std::fabs(std::sin((x + 1.0) * 2.0 / 3.0 * __gnu_cxx::__math_constants<double>::__pi)) - 1.0 + ed(eg));
  }
#endif

  Eigen::Matrix<double,2,1> fpolar(double r, double th) {
    return Eigen::Matrix<double,2,1>(r * std::cos(th), r * std::sin(th));
  }


  //typedef Network<double,2,4,6,2> N_t;



  std::size_t compute_ival(std::size_t e) {
    if (e < 10) return 1;
    if (e < 40) return 2;
    if (e < 100) return 10;
    return 100;
  }


  template<typename _Float>
  struct rgb {
    _Float r;
    _Float g;
    _Float b;
  };


  template<bool, typename _Float>
  rgb<_Float> to_color(const Eigen::Matrix<_Float,1,1>& an) {
    return rgb<_Float> { std:max(_Float(0), _Float(1) - _Float(2) * an[0]),
			 _Float(0),
			 std::max(_Float(0), _Float(2) * (an[0] - _Float(0.5)))
    };
  }

  template<bool, typename _Float>
  rgb<_Float> to_color(const Eigen::Matrix<_Float,2,1>& an) {
    return rgb<_Float> { an[0], _Float(0), an[1] };
  }

  template<bool, typename _Float>
  rgb<_Float> to_color(const Eigen::Matrix<_Float,3,1>& an) {
    return rgb<_Float> { an[0], an[2], an[1] };
  }

  template<bool _Normalize, typename _Float>
  rgb<_Float> to_color(const Eigen::Matrix<_Float,4,1>& an) {
    Eigen::Matrix<_Float,3,4> colmat;
    colmat <<
      1.0, 0.5, 0.0, 0.5,
      0.0, 1.0, 1.0, 0.0,
      0.0, 0.0, 1.0, 1.0;
    Eigen::Matrix<_Float,3,1> res = colmat * an;
    if (_Normalize)
      res /= res.norm();

    return rgb<_Float> { std::min(res[0], _Float(1)),
	std::min(res[1], _Float(1)),
	std::min(res[2], _Float(1)) };
  }

  template<bool _Normalize, typename _Float>
  rgb<_Float> to_color(const Eigen::Matrix<_Float,Eigen::Dynamic,1>& an) {
    if (an.rows() == 6) {
      Eigen::Matrix<_Float,3,6> colmat;
      colmat <<
	1.0, 1.0, 0.0, 0.0, 0.0, 1.0,
	0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
	0.0, 0.0, 0.0, 1.0, 1.0, 1.0;
      Eigen::Matrix<_Float,3,1> res = colmat * an;
      if (_Normalize)
	res /= res.norm();

      return rgb<_Float> { res[0], res[1], res[2] };
    }
    abort();
  }



  constexpr unsigned ndecisive_levels = 5;
  constexpr double decision_level_steps = 1.0 / double(ndecisive_levels);

  template<typename _Float>
  unsigned decisive_level(const Eigen::Matrix<_Float,1,1>& an) {
    return an[0] / decision_level_steps;
  }

  template<typename _Float>
  unsigned decisive_level(const Eigen::Matrix<_Float,2,1>& an) {
    return std::abs(an[0] - an[1]) / decision_level_steps;
  }

  template<typename _Float>
  unsigned decisive_level(const Eigen::Matrix<_Float,3,1>& an) {
    auto vmin = an.minCoeff();
    auto vmax = an.maxCoeff();
    auto vmed = an.sum() - vmin - vmax;
    return (vmax - vmed) / decision_level_steps;
  }

  template<typename _Float>
  unsigned decisive_level(const Eigen::Matrix<_Float,4,1>& an) {
    std::array<_Float,4> tmp { an[0], an[1], an[2], an[3] };
    std::sort(tmp.begin(), tmp.end());
    return (tmp[3] - tmp[2]) / decision_level_steps;
  }

  template<typename _Float>
  unsigned decisive_level(const Eigen::Matrix<_Float,Eigen::Dynamic,1>& an) {
    const auto n = an.rows();
    std::vector<_Float> tmp(n);
    for (size_t i = 0; i < n; ++i)
      tmp[i] = an[i];
    std::sort(tmp.begin(), tmp.end());
    return (tmp[n - 1] - tmp[n - 2]) / decision_level_steps;
  }


  void decisive_color(const Cairo::RefPtr<Cairo::Context>& cr,
		      unsigned lvl1, unsigned lvl2) {
    if (lvl1 > lvl2)
      std::swap(lvl1, lvl2);
    auto col = decision_level_steps * lvl2;
    cr->set_source_rgb(col, col, col);
  }


} // namespace


template<class _NN, unsigned _Level = _NN::nlayers, bool _Normalize = true>
class ResultArea : public Gtk::DrawingArea {
public:
  ResultArea(_NN& nn_, const train_data<_NN>& td_, bool show_data_ = true,
	     double amp_factor_ = 1.0, double diameter_ = 16.0)
    : Gtk::DrawingArea(), nn(nn_), td(td_), show_data(show_data_),
      amp_factor(amp_factor_), diameter(diameter_) { }
  virtual ~ResultArea() { }

protected:
  virtual bool on_draw(const Cairo::RefPtr<Cairo::Context>& cr);

private:
  _NN& nn;
  const train_data<_NN>& td;
  bool show_data;
  double amp_factor;
  double diameter;
};


template<class _NN, unsigned _Level, bool _Normalize>
bool ResultArea<_NN,_Level,_Normalize>::on_draw(const Cairo::RefPtr<Cairo::Context>& cr) {
  typedef typename _NN::float_type float_type;

  Gtk::Allocation allocation = get_allocation();
  const int width = allocation.get_width();
  const int all_height = allocation.get_height();
  const int height = all_height - (show_data ? 0 : 10);

#if 0
  float_type xmin = -1.0;
  float_type xmax = 1.0;
  float_type ymin = -1.0;
  float_type ymax = 0.5;
#else
  float_type xmin = -1.0;
  float_type xmax = 1.0;
  float_type ymin = -1.0;
  float_type ymax = 1.0;
#endif

  cr->scale(width / (xmax - xmin), -height / (ymax - ymin));
  cr->translate(-xmin, -ymax);
  cr->set_line_width(1.0 / std::min(width / (xmax - xmin), height / (ymax - ymin)));

  cr->set_source_rgb(0.0, 0.0, 0.0);
  cr->paint();

  cr->set_source_rgb(1.0, 1.0, 1.0);
  cr->move_to(xmin, 0.0);
  cr->line_to(xmax, 0.0);
  cr->stroke();

  cr->move_to(0.0, ymin);
  cr->line_to(0.0, ymax);
  cr->stroke();

  float_type d = diameter / float_type(2) / width;

  const float_type step = diameter / width;
  for (float_type y = ymin + step / float_type(2); y < ymax; y += step)
    for (float_type x = xmin + step / float_type(2); x < xmax; x += step) {
      typename _NN::a0_t in = { x, y };
      auto res = eval<_Level>(nn, in);
      auto col = to_color<_Normalize>(res);
      if (amp_factor == 2.0)
	cr->set_source_rgb(col.r * col.r, col.g * col.g, col.b * col.b);
      else if (amp_factor == 1.0)
	cr->set_source_rgb(col.r, col.g, col.b);
      else
	cr->set_source_rgb(std::min(1.0, std::pow(col.r, amp_factor)),
			   std::min(1.0, std::pow(col.g, amp_factor)),
			   std::min(1.0, std::pow(col.b, amp_factor)));

      cr->arc(x, y, d, 0, 2 * __gnu_cxx::__math_constants<float_type>::__pi);
      cr->fill();

      auto res_decisive = decisive_level(res);

      if (y > ymin + step / 2.0) {
	typename _NN::a0_t in2 = { x, y - step };
	auto res2 = eval<_Level>(nn, in2);
	auto res_decisive2 = decisive_level(res2);
	if (res_decisive != res_decisive2) {
	  cr->set_source_rgb(0.0, 0.0, 0.0);
	  cr->set_line_width(3.0 / std::min(width / (xmax - xmin), height / (ymax - ymin)));
	  cr->move_to(x - step / 2.0, y - step / 2.0);
	  cr->line_to(x + step / 2.0, y - step / 2.0);
	  cr->stroke_preserve();
	  decisive_color(cr, res_decisive, res_decisive2);
	  cr->set_line_width(1.0 / std::min(width / (xmax - xmin), height / (ymax - ymin)));
	  cr->stroke();
	}
      }

      if (x > xmin + step / 2.0) {
	typename _NN::a0_t in2 = { x - step, y };
	auto res2 = eval<_Level>(nn, in2);
	auto res_decisive2 = decisive_level(res2);
	if (res_decisive != res_decisive2) {
	  cr->set_source_rgb(0.0, 0.0, 0.0);
	  cr->set_line_width(3.0 / std::min(width / (xmax - xmin), height / (ymax - ymin)));
	  cr->move_to(x - step / 2.0, y - step / 2.0);
	  cr->line_to(x - step / 2.0, y + step / 2.0);
	  cr->stroke_preserve();
	  decisive_color(cr, res_decisive, res_decisive2);
	  cr->set_line_width(1.0 / std::min(width / (xmax - xmin), height / (ymax - ymin)));
	  cr->stroke();
	}
      }
    }

  if (show_data) {
    cr->set_source_rgb(255,255,255);

    auto r0 = td.template cbegin<0>();
    while (r0 != td.template cend<0>()) {
      auto x = (*r0)[0];
      auto y = (*r0)[1];
      cr->move_to(x-d, y);
      cr->line_to(x+d, y);
      cr->stroke();
      cr->move_to(x, y-d);
      cr->line_to(x, y+d);
      cr->stroke();
      ++r0;
    }

    auto r1 = td.template cbegin<1>();
    while (r1 != td.template cend<1>()) {
      auto x = (*r1)[0];
      auto y = (*r1)[1];
      cr->move_to(x-d, y-d);
      cr->line_to(x+d, y+d);
      cr->stroke();
      cr->move_to(x+d, y-d);
      cr->line_to(x-d, y+d);
      cr->stroke();
      ++r1;
    }

    if (td.nclasses > 2) {
      auto r2 = td.template cbegin<2>();
      while (r2 != td.template cend<2>()) {
	auto x = (*r2)[0];
	auto y = (*r2)[1];
	cr->arc(x, y, d,
		float_type(0), 2 * __gnu_cxx::__math_constants<float_type>::__pi);
	cr->stroke();
	++r2;
      }
    }
  } else {
    typename _NN::a0_t in = { 0, 0 };
    auto v = eval<_Level>(nn, in);
    unsigned n = v.rows();
    const double xstep = (xmax - xmin) / n;

    cr->rectangle(xmin, ymin, xstep, -0.2);
    v[0] = 1;
    v[1] = 0;
    if (n > 2) {
      v[2] = 0;
      if (n > 3) {
	v[3] = 0;
	if (n > 4) {
	  v[4] = 0;
	  if (n > 5) {
	    v[5] = 0;
	  }
	}
      }
    }
    auto col = to_color<_Normalize>(v);
    cr->set_source_rgb(col.r * col.r, col.g * col.g, col.b * col.b);
    cr->fill();

    cr->rectangle(xmin + xstep, ymin, xstep, -0.2);
    v[0] = 0;
    v[1] = 1;
    if (n > 2) {
      v[2] = 0;
      if (n > 3) {
	v[3] = 0;
	if (n > 4) {
	  v[4] = 0;
	  if (n > 5) {
	    v[5] = 0;
	  }
	}
      }
    }
    col = to_color<_Normalize>(v);
    cr->set_source_rgb(col.r * col.r, col.g * col.g, col.b * col.b);
    cr->fill();

    if (n > 2) {
      cr->rectangle(xmin + 2 * xstep, ymin, xstep, -0.2);
      v[0] = 0;
      v[1] = 0;
      if (n > 2) {
	v[2] = 1;
	if (n > 3) {
	  v[3] = 0;
	  if (n > 4) {
	    v[4] = 0;
	    if (n > 5) {
	      v[5] = 0;
	    }
	  }
	}
      }
      col = to_color<_Normalize>(v);
      cr->set_source_rgb(col.r * col.r, col.g * col.g, col.b * col.b);
      cr->fill();

      if (n > 3) {
	cr->rectangle(xmin + 3 * xstep, ymin, xstep, -0.2);
	v[0] = 0;
	v[1] = 0;
	if (n > 2) {
	  v[2] = 0;
	  if (n > 3) {
	    v[3] = 1;
	    if (n > 4) {
	      v[4] = 0;
	      if (n > 5) {
		v[5] = 0;
	      }
	    }
	  }
	}
	col = to_color<_Normalize>(v);
	cr->set_source_rgb(col.r * col.r, col.g * col.g, col.b * col.b);
	cr->fill();

	if (n > 4) {
	  cr->rectangle(xmin + 4 * xstep, ymin, xstep, -0.2);
	  v[0] = 0;
	  v[1] = 0;
	  if (n > 2) {
	    v[2] = 0;
	    if (n > 3) {
	      v[3] = 0;
	      if (n > 4) {
		v[4] = 1;
		if (n > 5) {
		  v[5] = 0;
		}
	      }
	    }
	  }
	  col = to_color<_Normalize>(v);
	  cr->set_source_rgb(col.r * col.r, col.g * col.g, col.b * col.b);
	  cr->fill();

	  if (n > 5) {
	    cr->rectangle(xmin + 5 * xstep, ymin, xstep, -0.2);
	    v[0] = 0;
	    v[1] = 0;
	    if (n > 2) {
	      v[2] = 0;
	      if (n > 3) {
		v[3] = 0;
		if (n > 4) {
		  v[4] = 0;
		  if (n > 5) {
		    v[5] = 1;
		  }
		}
	      }
	    }
	    col = to_color<_Normalize>(v);
	    cr->set_source_rgb(col.r * col.r, col.g * col.g, col.b * col.b);
	    cr->fill();
	  }
	}
      }
    }
  }

  return true;
}


template<class _NN>
class GraphArea : public Gtk::DrawingArea {
public:
  GraphArea(_NN& nn_, int selected)
    : Gtk::DrawingArea(), nn(nn_) {
    for (std::size_t i = 0; i <= nn.nlayers; ++i) {
      std::size_t nnodes = nn.get_nnodes(i);
      for (std::size_t j = 0; j < nnodes; ++j)
	graphinfo[std::make_pair(i, j)] =
	  (graphnodeinfo_type) { 0, 0, i == selected };
    }

    add_events(Gdk::BUTTON_PRESS_MASK);
    signal_event().connect(sigc::mem_fun(*this, &GraphArea::on_event_handler));
  }
  virtual ~GraphArea() { }

  void do_select(int n) {
    std::for_each(graphinfo.begin(), graphinfo.end(),
		  [n](std::pair<const graphnodeidx_type, graphnodeinfo_type>& p) {
		    std::get<1>(p).selected = std::get<0>(std::get<0>(p)) == n;
		  });
  }

protected:
  virtual bool on_draw(const Cairo::RefPtr<Cairo::Context>& cr);

  bool on_event_handler(GdkEvent* event) {
    if (event->type == GDK_BUTTON_PRESS) {

    }
  }

private:
  _NN& nn;

  typedef std::pair<std::size_t, std::size_t> graphnodeidx_type;
  struct graphnodeinfo_type {
    std::size_t x;
    std::size_t y;
    bool selected;
  };
  std::map<graphnodeidx_type, graphnodeinfo_type> graphinfo;

  static constexpr int xinc = 60;
  static constexpr int yinc = 20;

  static constexpr int radius = 8;
};


template<class _NN>
bool GraphArea<_NN>::on_draw(const Cairo::RefPtr<Cairo::Context>& cr) {
  typedef typename _NN::float_type float_type;

  Gtk::Allocation allocation = get_allocation();
  const int width = allocation.get_width();
  const int height = allocation.get_height();

  auto nlayers = nn.nlayers;
  const int xdist_centers = xinc * nlayers;
  const int xpad = (width - xdist_centers) / 2;

  cr->set_line_width(1.0);
  cr->set_source_rgb(0.0, 0.0, 0.0);
  for (std::size_t i = 0; i < nlayers; ++i) {
    std::size_t nnodes_from = nn.get_nnodes(i);
    std::size_t nnodes_to = nn.get_nnodes(i + 1);

    const int ydist_centers_from = yinc * (nnodes_from - 1);
    const int ypad_from = (height - ydist_centers_from) / 2;

    const int ydist_centers_to = yinc * (nnodes_to - 1);
    const int ypad_to = (height - ydist_centers_to) / 2;

    for (std::size_t j = 0; j < nnodes_from; ++j)
      for (std::size_t k = 0; k < nnodes_to; ++k) {
	cr->move_to(xpad + xinc * i, ypad_from + yinc * j);
	cr->line_to(xpad + xinc * (i + 1), ypad_to + yinc * k);
	cr->stroke();
      }
  }

  cr->set_line_width(2.0);
  for (std::size_t i = 0; i <= nlayers; ++i) {
    std::size_t nnodes = nn.get_nnodes(i);

    const int ydist_centers = yinc * (nnodes - 1);
    const int ypad = (height - ydist_centers) / 2;

    for (std::size_t j = 0; j < nnodes; ++j) {
      auto ni = graphinfo.find(std::make_pair(i, j));
      if (ni == graphinfo.end())
	abort();
      std::get<1>(*ni).x = xpad + xinc * i;
      std::get<1>(*ni).y = ypad + yinc * j;

      cr->arc(xpad + xinc * i, ypad + yinc * j, radius, 0, 2.0 * __gnu_cxx::__math_constants<float_type>::__pi);
      if (std::get<1>(*ni).selected)
	cr->set_source_rgb(1.0, 0.65, 0.0);
      else
	cr->set_source_rgb(1.0, 1.0, 1.0);
      cr->fill_preserve();
      cr->set_source_rgb(0.0, 0.0, 0.0);
      cr->stroke();
    }
  }
}


template<class _NN>
class AppWindow : public Gtk::Window {
public:
  AppWindow(_NN&, const train_data<_NN>&);

private:
  Gtk::HBox all;
  Gtk::VBox small;
  ResultArea<_NN, 1, false> resultsmall1;
  // ResultArea<_NN, 2, false> resultsmall2;
  ResultArea<_NN> resultsmallN;
  Gtk::Box empty;
  Gtk::VBox mainbox;
  Gtk::Box full;
  ResultArea<_NN, 1, false>* result1;
  ResultArea<_NN, 2, false>* result2;
  ResultArea<_NN>* resultN;
  Gtk::DrawingArea* result;
  Gtk::HBox info;
  GraphArea<_NN> graph;
  Gtk::ButtonBox control;
  Gtk::Button next1;
  Gtk::Button next10;
  Gtk::Button next100;
  Gtk::VBox extra;
  Gtk::Label label;
  Gtk::Button quit_button;

  void redraw();

  bool switch1(GdkEvent* event) {
    if (event->type == GDK_BUTTON_PRESS)
      do_switch(result1, 1);
  }
  bool switch2(GdkEvent* event) {
    if (event->type == GDK_BUTTON_PRESS)
      do_switch(result2, 2);
  }
  bool switchN(GdkEvent* event) {
    if (event->type == GDK_BUTTON_PRESS)
      do_switch(resultN, 2/*3*/);
  }
  void do_switch(Gtk::DrawingArea* newp, int n) {
    if (result != newp) {
      full.remove(*result);
      result = newp;
      full.pack_start(*result);
      result->show();
      result->queue_draw();

      graph.do_select(n);
      graph.queue_draw();
    }
  }

  void adv1() { nn << td; redraw(); }
  void adv10() { nn << 10 * td; redraw(); }
  void adv100() { nn << 100 * td; redraw(); }

  _NN& nn;
  const train_data<_NN>& td;
  int selected;
};


template<class _NN>
AppWindow<_NN>::AppWindow(_NN& nn_, const train_data<_NN>& td_)
  : Gtk::Window(),
    all(),
    small(Gtk::ORIENTATION_VERTICAL),
    resultsmall1(nn_, td_, false, 1.0, 8.0),
    // resultsmall2(nn_, td_, false, 1.0, 8.0),
    resultsmallN(nn_, td_, false, 2.0, 8.0),
    result1(new ResultArea<_NN, 1, false>(nn_, td_, true, 1.0, 16.0)),
    result2(new ResultArea<_NN, 2, false>(nn_, td_, true, 1.0, 16.0)),
    resultN(new ResultArea<_NN>(nn_, td_, true, 2.0, 16.0)),
    result(resultN),
    info(),
    selected(nn_.nlayers),
    graph(nn_, nn_.nlayers),
    control(Gtk::ORIENTATION_VERTICAL),
    label("test"),
    next1("+1"),
    next10("+10"),
    next100("+100"),
    quit_button("Exit"),
    nn(nn_),
    td(td_) {
  Glib::RefPtr<Gdk::Screen> screen = Gdk::Screen::get_default();
#if 0
  int width = std::min(screen->get_width() - 50, 900);
  int height = std::min(screen->get_height() - 50, 600);
#else
  int width = std::min(screen->get_width() - 50, 800);
  int height = std::min(screen->get_height() - 50, 800);
#endif
  set_default_size(width, height);
  set_title("NN BP Output");

  all.set_homogeneous(false);
#if 0
  resultsmall1.set_size_request(120, 90);
#else
  resultsmall1.set_size_request(120, 120);
#endif
  small.pack_start(resultsmall1, Gtk::PACK_SHRINK, 10);
  // resultsmall2.set_size_request(120, 90);
  // small.pack_start(resultsmall2, Gtk::PACK_SHRINK, 10);
#if 0
  resultsmallN.set_size_request(120, 90);
#else
  resultsmallN.set_size_request(120, 120);
#endif
  small.pack_start(resultsmallN, Gtk::PACK_SHRINK, 10);
  small.pack_start(empty, Gtk::PACK_EXPAND_WIDGET);
  all.pack_start(small, Gtk::PACK_SHRINK, 10);

  resultsmall1.add_events(Gdk::BUTTON_PRESS_MASK);
  resultsmall1.signal_event().connect(sigc::mem_fun(*this,
						    &AppWindow::switch1));
  // resultsmall2.add_events(Gdk::BUTTON_PRESS_MASK);
  // resultsmall2.signal_event().connect(sigc::mem_fun(*this,
  //						    &AppWindow::switch2));
  resultsmallN.add_events(Gdk::BUTTON_PRESS_MASK);
  resultsmallN.signal_event().connect(sigc::mem_fun(*this,
						    &AppWindow::switchN));

  full.pack_start(*result);
  mainbox.pack_start(full, Gtk::PACK_EXPAND_WIDGET);

  graph.set_size_request(-1, 120);
  info.pack_start(graph, Gtk::PACK_EXPAND_WIDGET);

  next1.signal_clicked().connect(sigc::mem_fun(*this, &AppWindow::adv1));
  control.pack_start(next1);
  next10.signal_clicked().connect(sigc::mem_fun(*this, &AppWindow::adv10));
  control.pack_start(next10);
  next100.signal_clicked().connect(sigc::mem_fun(*this, &AppWindow::adv100));
  control.pack_start(next100);
  info.pack_start(control, Gtk::PACK_SHRINK);

  std::ostringstream os;
  os << "Epoch " << nn.get_nepochs();
  label.set_label(os.str());
  extra.pack_start(label, Gtk::PACK_SHRINK);
  quit_button.signal_clicked().connect(sigc::mem_fun(*this, &AppWindow::hide));
  extra.pack_start(quit_button, Gtk::PACK_SHRINK);

  info.pack_start(extra, Gtk::PACK_SHRINK);

  mainbox.pack_start(info, Gtk::PACK_SHRINK);

  all.pack_start(mainbox, Gtk::PACK_EXPAND_WIDGET);

  add(all);
  show_all_children();
}


template<class _NN>
void AppWindow<_NN>::redraw() {
  std::ostringstream os;
  os << "Epoch " << nn.get_nepochs();
  label.set_label(os.str());

  result->queue_draw();
  resultsmall1.queue_draw();
  // resultsmall2.queue_draw();
  resultsmallN.queue_draw();
}

namespace {
  int on_cmd(const Glib::RefPtr<Gio::ApplicationCommandLine> &,
	     Glib::RefPtr<Gtk::Application> &app) {
    app->activate();
    return 0;
  }
}


int main(int argc, char *argv[]) {
  Glib::RefPtr<Gtk::Application> app
    = Gtk::Application::create(argc, argv, "org.akkadia.nn-bp.base",
			       Gio::APPLICATION_HANDLES_COMMAND_LINE);

  typedef double float_type;
  ed.param(std::normal_distribution<float_type>::param_type(0,
							    argc == 1
							    ? 0.1
							    : strtod(argv[1],
								     nullptr)));

  //typedef Network<float_type,2,4,6,2> N_t;
  // typedef Network<float_type,2,3,2> N_t;
  typedef Network<float_type,2,4,3> N_t;
  N_t nn(10);

  app->signal_command_line().connect(sigc::bind(sigc::ptr_fun(on_cmd), app), false);

#if 0
  __gnu_cxx::sfmt19937 e((std::random_device())());
  std::uniform_real_distribution<N_t::float_type> d(-1.0, 1.0);

  train_data<N_t> td(1000,
		     [](std::size_t n){return typename N_t::an_t { 1 - n, n }; },
		     [&d,&e]{return f1(d(e));},
		     [&d,&e]{return f2(d(e));});
#else
  __gnu_cxx::sfmt19937 e((std::random_device())());
  std::normal_distribution<N_t::float_type> r1(0.5, 1.0);
  std::normal_distribution<N_t::float_type> th1(__gnu_cxx::__math_constants<N_t::float_type>::__pi * 0.0,
						__gnu_cxx::__math_constants<N_t::float_type>::__pi * 0.02);
  std::normal_distribution<N_t::float_type> r2(0.5, 0.5);
  std::normal_distribution<N_t::float_type> th2(__gnu_cxx::__math_constants<N_t::float_type>::__pi * -2.0 / 3.0,
						__gnu_cxx::__math_constants<N_t::float_type>::__pi * 0.01);
  std::normal_distribution<N_t::float_type> r3(0.5, 1.0);
  std::normal_distribution<N_t::float_type> th3(__gnu_cxx::__math_constants<N_t::float_type>::__pi * 7.0 / 8.0,
						__gnu_cxx::__math_constants<N_t::float_type>::__pi * 0.02);

  train_data<N_t> td(1000,
		     [](std::size_t n){return typename N_t::an_t { n == 0 ? 1.0 : 0.0,
			   n == 1.0 ? 1 : 0.0, n == 2 ? 1.0 : 0.0 }; },
		     [&r1,&th1,&e]{return fpolar(r1(e), th1(e));},
		     [&r2,&th2,&e]{return fpolar(r2(e), th2(e));},
		     [&r3,&th3,&e]{return fpolar(r3(e), th3(e));});
#endif

  AppWindow<N_t> window(nn, td);

  return app->run(window);
}
