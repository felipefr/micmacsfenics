/*!
* \file   TFEL/Material/IsotropicLinearHardeningPlasticityIntegrationData.hxx
* \brief  this file implements the IsotropicLinearHardeningPlasticityIntegrationData class.
*         File generated by tfel version 4.1.0-dev
* \author Thomas Helfer
* \date   14 / 10 / 2016
 */

#ifndef LIB_TFELMATERIAL_ISOTROPICLINEARHARDENINGPLASTICITY_INTEGRATION_DATA_HXX
#define LIB_TFELMATERIAL_ISOTROPICLINEARHARDENINGPLASTICITY_INTEGRATION_DATA_HXX

#include<string>
#include<iostream>
#include<limits>
#include<stdexcept>
#include<algorithm>

#include"TFEL/Raise.hxx"
#include"TFEL/PhysicalConstants.hxx"
#include"TFEL/Config/TFELConfig.hxx"
#include"TFEL/Config/TFELTypes.hxx"
#include"TFEL/TypeTraits/IsFundamentalNumericType.hxx"
#include"TFEL/TypeTraits/IsScalar.hxx"
#include"TFEL/TypeTraits/IsReal.hxx"
#include"TFEL/TypeTraits/Promote.hxx"
#include"TFEL/Math/General/IEEE754.hxx"
#include"TFEL/Math/stensor.hxx"
#include"TFEL/Math/st2tost2.hxx"
#include "MFront/GenericBehaviour/State.hxx"
#include "MFront/GenericBehaviour/BehaviourData.hxx"
namespace tfel::material{

//! \brief forward declaration
template<ModellingHypothesis::Hypothesis hypothesis, typename NumericType, bool use_qt>
class IsotropicLinearHardeningPlasticityIntegrationData;

//! \brief forward declaration
template<ModellingHypothesis::Hypothesis hypothesis, typename NumericType>
std::ostream&
 operator <<(std::ostream&,const IsotropicLinearHardeningPlasticityIntegrationData<hypothesis, NumericType, false>&);

template<ModellingHypothesis::Hypothesis hypothesis, typename NumericType>
class IsotropicLinearHardeningPlasticityIntegrationData<hypothesis, NumericType, false>
{

static constexpr unsigned short N = ModellingHypothesisToSpaceDimension<hypothesis>::value;
static_assert(N==1||N==2||N==3);
static_assert(tfel::typetraits::IsFundamentalNumericType<NumericType>::cond);
static_assert(tfel::typetraits::IsReal<NumericType>::cond);

friend std::ostream& operator<< <>(std::ostream&,const IsotropicLinearHardeningPlasticityIntegrationData&);

static constexpr unsigned short TVectorSize = N;
typedef tfel::math::StensorDimeToSize<N> StensorDimeToSize;
static constexpr unsigned short StensorSize = StensorDimeToSize::value;
typedef tfel::math::TensorDimeToSize<N> TensorDimeToSize;
static constexpr unsigned short TensorSize = TensorDimeToSize::value;

using ushort =  unsigned short;
using Types = tfel::config::Types<N, NumericType, false>;
using Type = NumericType;
using real = typename Types::real;
using time = typename Types::time;
using length = typename Types::length;
using frequency = typename Types::frequency;
using speed = typename Types::speed;
using stress = typename Types::stress;
using strain = typename Types::strain;
using strainrate = typename Types::strainrate;
using stressrate = typename Types::stressrate;
using temperature = typename Types::temperature;
using thermalexpansion = typename Types::thermalexpansion;
using thermalconductivity = typename Types::thermalconductivity;
using massdensity = typename Types::massdensity;
using energydensity = typename Types::energydensity;
using TVector = typename Types::TVector;
using DisplacementTVector = typename Types::DisplacementTVector;
using ForceTVector = typename Types::ForceTVector;
using HeatFlux = typename Types::HeatFlux;
using TemperatureGradient = typename Types::TemperatureGradient;
using Stensor = typename Types::Stensor;
using StressStensor = typename Types::StressStensor;
using StressRateStensor = typename Types::StressRateStensor;
using StrainStensor = typename Types::StrainStensor;
using StrainRateStensor = typename Types::StrainRateStensor;
using FrequencyStensor = typename Types::FrequencyStensor;
using Tensor = typename Types::Tensor;
using DeformationGradientTensor = typename Types::DeformationGradientTensor;
using StressTensor = typename Types::StressTensor;
using StiffnessTensor = typename Types::StiffnessTensor;
using Stensor4 = typename Types::Stensor4;
using TangentOperator = StiffnessTensor;
using PhysicalConstants = tfel::PhysicalConstants<NumericType, false>;

protected: 

/*!
 * \brief eto increment
 */
StrainStensor deto;

/*!
 * \brief time increment
 */
time dt;

temperature dT;
public:

/*!
* \brief Default constructor
*/
IsotropicLinearHardeningPlasticityIntegrationData()
{}

/*!
* \brief Copy constructor
*/
IsotropicLinearHardeningPlasticityIntegrationData(const IsotropicLinearHardeningPlasticityIntegrationData& src)
: deto(src.deto),
dt(src.dt),
dT(src.dT)
{}

/*
 * \brief constructor for the Generic interface
 * \param[in] mgb_d: behaviour data
 */
IsotropicLinearHardeningPlasticityIntegrationData(const mfront::gb::BehaviourData& mgb_d)
: dt(mgb_d.dt),
dT(mgb_d.s1.external_state_variables[0]-mgb_d.s0.external_state_variables[0])
{
}


/*
* \brief scale the integration data by a scalar.
*/
template<typename Scal>
typename std::enable_if<
tfel::typetraits::IsFundamentalNumericType<Scal>::cond&&
tfel::typetraits::IsScalar<Scal>::cond&&
tfel::typetraits::IsReal<Scal>::cond&&
std::is_same<NumericType,typename tfel::typetraits::Promote<NumericType,Scal>::type>::value,
IsotropicLinearHardeningPlasticityIntegrationData&
>::type
scale(const IsotropicLinearHardeningPlasticityBehaviourData<hypothesis, NumericType, false>&, const Scal time_scaling_factor){
this->dt   *= time_scaling_factor;
this->deto *= time_scaling_factor;
this->dT *= time_scaling_factor;
return *this;
}

/*!
* \brief update the driving variable in case of substepping.
*/
IsotropicLinearHardeningPlasticityIntegrationData&
updateDrivingVariables(const IsotropicLinearHardeningPlasticityBehaviourData<hypothesis, NumericType, false>&){
return *this;
}

}; // end of IsotropicLinearHardeningPlasticityIntegrationDataclass

template<ModellingHypothesis::Hypothesis hypothesis, typename NumericType>
std::ostream&
operator <<(std::ostream& os,const IsotropicLinearHardeningPlasticityIntegrationData<hypothesis, NumericType, false>& b)
{
os << "Δεᵗᵒ : " << b.deto << '\n';
os << "σ : " << b.sig << '\n';
os << "Δt : " << b.dt << '\n';
os << "ΔT : " << b.dT << '\n';
return os;
}

} // end of namespace tfel::material

#endif /* LIB_TFELMATERIAL_ISOTROPICLINEARHARDENINGPLASTICITY_INTEGRATION_DATA_HXX */