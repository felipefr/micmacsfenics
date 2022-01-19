/*!
* \file   IsotropicLinearHardeningPlasticity-generic.cxx
* \brief  This file implements the umat interface for the IsotropicLinearHardeningPlasticity behaviour law
* \author Thomas Helfer
* \date   14 / 10 / 2016
*/

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif /* NOMINMAX */
#include <windows.h>
#ifdef small
#undef small
#endif /* small */
#endif /* _WIN32 */

#ifndef MFRONT_SHAREDOBJ
#define MFRONT_SHAREDOBJ TFEL_VISIBILITY_EXPORT
#endif /* MFRONT_SHAREDOBJ */

#include<iostream>
#include<cstdlib>
#include"TFEL/Material/OutOfBoundsPolicy.hxx"
#include"TFEL/Math/t2tot2.hxx"
#include"TFEL/Math/t2tost2.hxx"
#include"TFEL/Material/IsotropicLinearHardeningPlasticity.hxx"
#include"MFront/GenericBehaviour/Integrate.hxx"

#include"MFront/GenericBehaviour/IsotropicLinearHardeningPlasticity-generic.hxx"

static tfel::material::OutOfBoundsPolicy&
IsotropicLinearHardeningPlasticity_getOutOfBoundsPolicy(){
using namespace tfel::material;
static OutOfBoundsPolicy policy = None;
return policy;
}

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

MFRONT_SHAREDOBJ const char* 
IsotropicLinearHardeningPlasticity_build_id = "";

MFRONT_SHAREDOBJ const char* 
IsotropicLinearHardeningPlasticity_mfront_ept = "IsotropicLinearHardeningPlasticity";

MFRONT_SHAREDOBJ const char* 
IsotropicLinearHardeningPlasticity_tfel_version = "4.1.0-dev";

MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_mfront_mkt = 1u;

MFRONT_SHAREDOBJ const char *
IsotropicLinearHardeningPlasticity_mfront_interface = "Generic";

MFRONT_SHAREDOBJ const char *
IsotropicLinearHardeningPlasticity_src = "IsotropicLinearHardeningPlasticity.mfront";

MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_nModellingHypotheses = 5u;

MFRONT_SHAREDOBJ const char * 
IsotropicLinearHardeningPlasticity_ModellingHypotheses[5u] = {"AxisymmetricalGeneralisedPlaneStrain",
"Axisymmetrical",
"PlaneStrain",
"GeneralisedPlaneStrain",
"Tridimensional"};

MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_nMainVariables = 1;
MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_nGradients = 1;

MFRONT_SHAREDOBJ int IsotropicLinearHardeningPlasticity_GradientsTypes[1] = {1};
MFRONT_SHAREDOBJ const char * IsotropicLinearHardeningPlasticity_Gradients[1] = {"Strain"};
MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_nThermodynamicForces = 1;

MFRONT_SHAREDOBJ int IsotropicLinearHardeningPlasticity_ThermodynamicForcesTypes[1] = {1};
MFRONT_SHAREDOBJ const char * IsotropicLinearHardeningPlasticity_ThermodynamicForces[1] = {"Stress"};
MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_nTangentOperatorBlocks = 2;

MFRONT_SHAREDOBJ const char * IsotropicLinearHardeningPlasticity_TangentOperatorBlocks[2] = {"Stress",
"Strain"};
MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_BehaviourType = 1u;

MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_BehaviourKinematic = 1u;

MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_SymmetryType = 0u;

MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_ElasticSymmetryType = 0u;

MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_api_version = 1u;

MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_TemperatureRemovedFromExternalStateVariables = 1u;
MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_UsableInPurelyImplicitResolution = 1;

MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_nMaterialProperties = 4u;

MFRONT_SHAREDOBJ const char *IsotropicLinearHardeningPlasticity_MaterialProperties[4u] = {"YoungModulus",
"PoissonRatio",
"HardeningSlope",
"YieldStrength"};

MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_nInternalStateVariables = 2;
MFRONT_SHAREDOBJ const char * IsotropicLinearHardeningPlasticity_InternalStateVariables[2] = {"ElasticStrain",
"EquivalentPlasticStrain"};
MFRONT_SHAREDOBJ int IsotropicLinearHardeningPlasticity_InternalStateVariablesTypes [] = {1,0};

MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_nExternalStateVariables = 0;
MFRONT_SHAREDOBJ const char * const * IsotropicLinearHardeningPlasticity_ExternalStateVariables = nullptr;

MFRONT_SHAREDOBJ const int * IsotropicLinearHardeningPlasticity_ExternalStateVariablesTypes  = nullptr;

MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_nParameters = 2;
MFRONT_SHAREDOBJ const char * IsotropicLinearHardeningPlasticity_Parameters[2] = {"minimal_time_step_scaling_factor",
"maximal_time_step_scaling_factor"};
MFRONT_SHAREDOBJ int IsotropicLinearHardeningPlasticity_ParametersTypes [] = {0,0};

MFRONT_SHAREDOBJ double IsotropicLinearHardeningPlasticity_minimal_time_step_scaling_factor_ParameterDefaultValue = 0.1;

MFRONT_SHAREDOBJ double IsotropicLinearHardeningPlasticity_maximal_time_step_scaling_factor_ParameterDefaultValue = 1.7976931348623e+308;

MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_requiresStiffnessTensor = 0;
MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_requiresThermalExpansionCoefficientTensor = 0;
MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_nInitializeFunctions= 0;

MFRONT_SHAREDOBJ const char * const * IsotropicLinearHardeningPlasticity_InitializeFunctions = nullptr;


MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_ComputesInternalEnergy = 0;

MFRONT_SHAREDOBJ unsigned short IsotropicLinearHardeningPlasticity_ComputesDissipatedEnergy = 0;

MFRONT_SHAREDOBJ void
IsotropicLinearHardeningPlasticity_setOutOfBoundsPolicy(const int p){
if(p==0){
IsotropicLinearHardeningPlasticity_getOutOfBoundsPolicy() = tfel::material::None;
} else if(p==1){
IsotropicLinearHardeningPlasticity_getOutOfBoundsPolicy() = tfel::material::Warning;
} else if(p==2){
IsotropicLinearHardeningPlasticity_getOutOfBoundsPolicy() = tfel::material::Strict;
} else {
std::cerr << "IsotropicLinearHardeningPlasticity_setOutOfBoundsPolicy: invalid argument\n";
}
}

MFRONT_SHAREDOBJ int
IsotropicLinearHardeningPlasticity_setParameter(const char *const key,const double value){
using tfel::material::IsotropicLinearHardeningPlasticityParametersInitializer;
auto& i = IsotropicLinearHardeningPlasticityParametersInitializer::get();
try{
i.set(key,value);
} catch(std::runtime_error& e){
std::cerr << e.what() << std::endl;
return 0;
}
return 1;
}

MFRONT_SHAREDOBJ int IsotropicLinearHardeningPlasticity_AxisymmetricalGeneralisedPlaneStrain(mfront_gb_BehaviourData* const d){
using namespace tfel::material;
using real = mfront::gb::real;
constexpr auto h = ModellingHypothesis::AXISYMMETRICALGENERALISEDPLANESTRAIN;
using Behaviour = IsotropicLinearHardeningPlasticity<h,real,false>;
const auto r = mfront::gb::integrate<Behaviour>(*d,Behaviour::STANDARDTANGENTOPERATOR, IsotropicLinearHardeningPlasticity_getOutOfBoundsPolicy());
return r;
} // end of IsotropicLinearHardeningPlasticity_AxisymmetricalGeneralisedPlaneStrain

MFRONT_SHAREDOBJ int IsotropicLinearHardeningPlasticity_Axisymmetrical(mfront_gb_BehaviourData* const d){
using namespace tfel::material;
using real = mfront::gb::real;
constexpr auto h = ModellingHypothesis::AXISYMMETRICAL;
using Behaviour = IsotropicLinearHardeningPlasticity<h,real,false>;
const auto r = mfront::gb::integrate<Behaviour>(*d,Behaviour::STANDARDTANGENTOPERATOR, IsotropicLinearHardeningPlasticity_getOutOfBoundsPolicy());
return r;
} // end of IsotropicLinearHardeningPlasticity_Axisymmetrical

MFRONT_SHAREDOBJ int IsotropicLinearHardeningPlasticity_PlaneStrain(mfront_gb_BehaviourData* const d){
using namespace tfel::material;
using real = mfront::gb::real;
constexpr auto h = ModellingHypothesis::PLANESTRAIN;
using Behaviour = IsotropicLinearHardeningPlasticity<h,real,false>;
const auto r = mfront::gb::integrate<Behaviour>(*d,Behaviour::STANDARDTANGENTOPERATOR, IsotropicLinearHardeningPlasticity_getOutOfBoundsPolicy());
return r;
} // end of IsotropicLinearHardeningPlasticity_PlaneStrain

MFRONT_SHAREDOBJ int IsotropicLinearHardeningPlasticity_GeneralisedPlaneStrain(mfront_gb_BehaviourData* const d){
using namespace tfel::material;
using real = mfront::gb::real;
constexpr auto h = ModellingHypothesis::GENERALISEDPLANESTRAIN;
using Behaviour = IsotropicLinearHardeningPlasticity<h,real,false>;
const auto r = mfront::gb::integrate<Behaviour>(*d,Behaviour::STANDARDTANGENTOPERATOR, IsotropicLinearHardeningPlasticity_getOutOfBoundsPolicy());
return r;
} // end of IsotropicLinearHardeningPlasticity_GeneralisedPlaneStrain

MFRONT_SHAREDOBJ int IsotropicLinearHardeningPlasticity_Tridimensional(mfront_gb_BehaviourData* const d){
using namespace tfel::material;
using real = mfront::gb::real;
constexpr auto h = ModellingHypothesis::TRIDIMENSIONAL;
using Behaviour = IsotropicLinearHardeningPlasticity<h,real,false>;
const auto r = mfront::gb::integrate<Behaviour>(*d,Behaviour::STANDARDTANGENTOPERATOR, IsotropicLinearHardeningPlasticity_getOutOfBoundsPolicy());
return r;
} // end of IsotropicLinearHardeningPlasticity_Tridimensional

#ifdef __cplusplus
}
#endif /* __cplusplus */

