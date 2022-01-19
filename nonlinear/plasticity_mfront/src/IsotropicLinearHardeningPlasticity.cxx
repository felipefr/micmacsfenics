/*!
* \file   IsotropicLinearHardeningPlasticity.cxx
* \brief  this file implements the IsotropicLinearHardeningPlasticity Behaviour.
*         File generated by tfel version 4.1.0-dev
* \author Thomas Helfer
* \date   14 / 10 / 2016
 */

#include<string>
#include<cstring>
#include<sstream>
#include<fstream>
#include<stdexcept>

#include"TFEL/Raise.hxx"
#include"TFEL/Material/IsotropicLinearHardeningPlasticityBehaviourData.hxx"
#include"TFEL/Material/IsotropicLinearHardeningPlasticityIntegrationData.hxx"
#include"TFEL/Material/IsotropicLinearHardeningPlasticity.hxx"

namespace tfel::material{

IsotropicLinearHardeningPlasticityParametersInitializer&
IsotropicLinearHardeningPlasticityParametersInitializer::get()
{
static IsotropicLinearHardeningPlasticityParametersInitializer i;
return i;
}

IsotropicLinearHardeningPlasticityParametersInitializer::IsotropicLinearHardeningPlasticityParametersInitializer()
{
this->minimal_time_step_scaling_factor = 0.1;
this->maximal_time_step_scaling_factor = 1.7976931348623e+308;
// Reading parameters from a file
IsotropicLinearHardeningPlasticityParametersInitializer::readParameters(*this,"IsotropicLinearHardeningPlasticity-parameters.txt");
}

void
IsotropicLinearHardeningPlasticityParametersInitializer::set(const char* const key,
const double v){
using namespace std;
if(::strcmp("minimal_time_step_scaling_factor",key)==0){
this->minimal_time_step_scaling_factor = v;
} else if(::strcmp("maximal_time_step_scaling_factor",key)==0){
this->maximal_time_step_scaling_factor = v;
} else {
tfel::raise("IsotropicLinearHardeningPlasticityParametersInitializer::set: "
" no parameter named '"+std::string(key)+"'");
}
}

double
IsotropicLinearHardeningPlasticityParametersInitializer::getDouble(const std::string& n,
const std::string& v)
{
double value;
std::istringstream converter(v);
converter >> value;
tfel::raise_if(!converter||(!converter.eof()),
"IsotropicLinearHardeningPlasticityParametersInitializer::getDouble: "
"can't convert '"+v+"' to double for parameter '"+ n+"'");
return value;
}

void
IsotropicLinearHardeningPlasticityParametersInitializer::readParameters(IsotropicLinearHardeningPlasticityParametersInitializer& pi,const char* const fn){
auto tokenize = [](const std::string& line){
std::istringstream tokenizer(line);
std::vector<std::string> tokens;
std::copy(std::istream_iterator<std::string>(tokenizer),
std::istream_iterator<std::string>(),
std::back_inserter(tokens));
return tokens;
};
std::ifstream f(fn);
if(!f){
return;
}
size_t ln = 1u;
while(!f.eof()){
auto line = std::string{};
std::getline(f,line);
auto tokens = tokenize(line);
auto throw_if = [ln,line,fn](const bool c,const std::string& m){
tfel::raise_if(c,"IsotropicLinearHardeningPlasticityParametersInitializer::readParameters: "
"error at line '"+std::to_string(ln)+"' "
"while reading parameter file '"+std::string(fn)+"'"
"("+m+")");
};
if(tokens.empty()){
continue;
}
if(tokens[0][0]=='#'){
continue;
}
throw_if(tokens.size()!=2u,"invalid number of tokens");
if("minimal_time_step_scaling_factor"==tokens[0]){
pi.minimal_time_step_scaling_factor = IsotropicLinearHardeningPlasticityParametersInitializer::getDouble(tokens[0],tokens[1]);
} else if("maximal_time_step_scaling_factor"==tokens[0]){
pi.maximal_time_step_scaling_factor = IsotropicLinearHardeningPlasticityParametersInitializer::getDouble(tokens[0],tokens[1]);
} else {
throw_if(true,"invalid parameter '"+tokens[0]+"'");
}
}
}

} // end of namespace tfel::material
