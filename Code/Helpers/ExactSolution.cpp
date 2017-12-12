
#ifndef ExactSolutionFlag_CPP
#define ExactSolutionFlag_CPP
#include <vector>
#include <string>
#include "ExactSolution.h"
#include "PointVal.cpp"


double ExactSolution::value (const Point<3> &p , const unsigned int component) const
{
  if(is_rectangular){
      const double delta = abs(mesh_points[0] -mesh_points[1]);
      if(abs(p(1))>mesh_points[0] || abs(p(0))>mesh_points[0]) {
        int ix = 0;
        int iy = 0;
        while(mesh_points[ix] > p(0)) ix++;
        while(mesh_points[ix] > p(1)) iy++;
        if(ix == 0 || iy == 0) {
          return 0.0;
        } else {
          double dx = (p(0) - mesh_points[ix])/delta;
          double dy = (p(1) - mesh_points[iy])/delta;
          double m1m1 = dx*dy;
          double m1p1 = dx*(1.0-dy);
          double p1p1 = (1.0-dx)*(1.0-dy);
          double p1m1 = (1.0-dx)*dy;

          switch (component) {
            case 0:
              return p1p1*vals[ix][iy].Ex.real() + p1m1*vals[ix][iy-1].Ex.real() + m1m1*vals[ix-1][iy-1].Ex.real() + m1p1*vals[ix-1][iy].Ex.real();
              break;
            case 1:
              return p1p1*vals[ix][iy].Ex.imag() + p1m1*vals[ix][iy-1].Ex.imag() + m1m1*vals[ix-1][iy-1].Ex.imag() + m1p1*vals[ix-1][iy].Ex.imag();
              break;
            case 2:
              return p1p1*vals[ix][iy].Ey.real() + p1m1*vals[ix][iy-1].Ey.real() + m1m1*vals[ix-1][iy-1].Ey.real() + m1p1*vals[ix-1][iy].Ey.real();
              break;
            case 3:
              return p1p1*vals[ix][iy].Ey.imag() + p1m1*vals[ix][iy-1].Ey.imag() + m1m1*vals[ix-1][iy-1].Ey.imag() + m1p1*vals[ix-1][iy].Ey.imag();
              break;
            case 4:
              return p1p1*vals[ix][iy].Ez.real() + p1m1*vals[ix][iy-1].Ez.real() + m1m1*vals[ix-1][iy-1].Ez.real() + m1p1*vals[ix-1][iy].Ez.real();
              break;
            case 5:
              return p1p1*vals[ix][iy].Ez.imag() + p1m1*vals[ix][iy-1].Ez.imag() + m1m1*vals[ix-1][iy-1].Ez.imag() + m1p1*vals[ix-1][iy].Ez.imag();
              break;
            default:
              return 0.0;
              break;
          }
        }
      } else {
        return 0.0;
      }
    }else {
    bool zero = false;
    if(p[0] > GlobalParams.M_R_XLength/2.0 - GlobalParams.M_BC_XPlus) zero = true;
    if(p[0] < -GlobalParams.M_R_XLength/2.0 + GlobalParams.M_BC_XMinus) zero = true;
    if(p[1] > GlobalParams.M_R_YLength/2.0 - GlobalParams.M_BC_YPlus) zero = true;
    if(p[1] < -GlobalParams.M_R_YLength/2.0 + GlobalParams.M_BC_YMinus) zero = true;
    if(p[2] > GlobalParams.M_R_ZLength/2.0) zero = true;
    if(zero){
      return 0;
    } else {
      return ModeMan.get_input_component( component, p, 0);
    }
  }

}


void ExactSolution::vector_value (const Point<3> &p,	Vector<double> &values) const
{
  if(is_rectangular){
    const double delta = abs(mesh_points[0] -mesh_points[1]);
    if(abs(p(1))>mesh_points[0] || abs(p(0))>mesh_points[0]) {
      int ix = 0;
      int iy = 0;
      while(mesh_points[ix] > p(0)) ix++;
      while(mesh_points[ix] > p(1)) iy++;
      if(ix == 0 || iy == 0) {
        for(int i = 0; i < values.size(); i++) {
          values[i] = 0.0;
        }
      } else {
        double dx = (p(0) - mesh_points[ix])/delta;
        double dy = (p(1) - mesh_points[iy])/delta;
        double m1m1 = dx*dy;
        double m1p1 = dx*(1.0-dy);
        double p1p1 = (1.0-dx)*(1.0-dy);
        double p1m1 = (1.0-dx)*dy;

        values[0] = p1p1*vals[ix][iy].Ex.real() + p1m1*vals[ix][iy-1].Ex.real() + m1m1*vals[ix-1][iy-1].Ex.real() + m1p1*vals[ix-1][iy].Ex.real();
        values[1] = p1p1*vals[ix][iy].Ex.imag() + p1m1*vals[ix][iy-1].Ex.imag() + m1m1*vals[ix-1][iy-1].Ex.imag() + m1p1*vals[ix-1][iy].Ex.imag();
        values[2] = p1p1*vals[ix][iy].Ey.real() + p1m1*vals[ix][iy-1].Ey.real() + m1m1*vals[ix-1][iy-1].Ey.real() + m1p1*vals[ix-1][iy].Ey.real();
        values[3] = p1p1*vals[ix][iy].Ey.imag() + p1m1*vals[ix][iy-1].Ey.imag() + m1m1*vals[ix-1][iy-1].Ey.imag() + m1p1*vals[ix-1][iy].Ey.imag();
        values[4] = p1p1*vals[ix][iy].Ez.real() + p1m1*vals[ix][iy-1].Ez.real() + m1m1*vals[ix-1][iy-1].Ez.real() + m1p1*vals[ix-1][iy].Ez.real();
        values[5] = p1p1*vals[ix][iy].Ez.imag() + p1m1*vals[ix][iy-1].Ez.imag() + m1m1*vals[ix-1][iy-1].Ez.imag() + m1p1*vals[ix-1][iy].Ez.imag();
      }
    } else {
      for(int i = 0; i < values.size(); i++) {
        values[i] = 0.0;
      }
    }
  } else {
    bool zero = false;
    if(p[0] > GlobalParams.M_R_XLength/2.0 - GlobalParams.M_BC_XPlus) zero = true;
    if(p[0] < -GlobalParams.M_R_XLength/2.0 + GlobalParams.M_BC_XMinus) zero = true;
    if(p[1] > GlobalParams.M_R_YLength/2.0 - GlobalParams.M_BC_YPlus) zero = true;
    if(p[1] < -GlobalParams.M_R_YLength/2.0 + GlobalParams.M_BC_YMinus) zero = true;
    if(p[2] > GlobalParams.M_R_ZLength/2.0) zero = true;
    if(zero) {
      for (unsigned int c=0; c<6; ++c) values[c] = 0.0;
    } else {
      for (unsigned int c=0; c<6; ++c) values[c] = ModeMan.get_input_component( c, p, 0);
    }
  }
}

std::vector<std::string> ExactSolution::split(std::string str,std::string sep) const{
    char* cstr=const_cast<char*>(str.c_str());
    char* current;
    std::vector<std::string> arr;
    current=strtok(cstr,sep.c_str());
    while(current!=NULL){
        arr.push_back(current);
        current=strtok(NULL,sep.c_str());
    }
    return arr;
}

double scientific_string_to_double(std::string inp) {
  std::istringstream os(inp);
  double d;
  os >> d;
  return d;
}

ExactSolution::ExactSolution(bool in_rectangular): Function<3>(6) {
  is_rectangular = in_rectangular;
  if(is_rectangular) {
    std::ifstream input( "filename.ext" );
    std::string line;
    getline( input, line );
    getline( input, line );
    float l_val =  3.0;
    for( ; getline( input, line ); )
    {
      std::vector<std::string> ls = split(line, "\t");
      float x = std::stof(ls[2]);
      if(x < l_val) {
        mesh_points.push_back(x);
        l_val = x;
      }
    }
    mesh_points.shrink_to_fit();
    unsigned int cnt  = mesh_points.size();
     vals = new PointVal*[cnt];
    for(int i = 0; i < cnt; i++) {
      vals[i] = new PointVal[cnt];
    }

    std::ifstream input2( "filename.ext" );
    std::string line2;
    getline( input2, line2 );
    getline( input2, line2 );
    for (int i = 0; i < cnt ; ++ i) {
      for (int j = 0; j < cnt ; ++ j) {
        getline( input2, line2 );
        std::vector<std::string> ls = split(line, "\t");
        vals[j][i].set(scientific_string_to_double(ls[5]),scientific_string_to_double(ls[4]),scientific_string_to_double(ls[3]),scientific_string_to_double(ls[8]),scientific_string_to_double(ls[7]),scientific_string_to_double(ls[6]));
      }
    }

  }
}

#endif
