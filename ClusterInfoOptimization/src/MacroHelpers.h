//
// Copyright (C) 2002-2023 CERN for the benefit of the ATLAS collaboration
//
// Dear emacs, this is -*- c++ -*-
//

#ifndef CALORECGPU_MACROHELPERS_H
#define CALORECGPU_MACROHELPERS_H

/** @file MacroHelpers.h
 *  Contains some helpful macros to help with repetitive code...
 */

#define CRGPU_EMPTY()

/** @brief Helper macro, defers macro expansion to the next level of evaluation to allow recursive macros.
 */
#define CRGPU_DEFER(...) __VA_ARGS__ CRGPU_EMPTY()

/** @brief Helper macro, takes the arguments and expands to nothing.
 */

#define CRGPU_IGNORE(...)

/** @brief Helper macro, concatenates two arguments
 */
#define CRGPU_CONCAT(X, Y) CRGPU_CONCAT_HELPER(X, Y)

/** @brief Helper macro for concatenation.
 */
#define CRGPU_CONCAT_HELPER(X,...) X ## __VA_ARGS__


/** @brief Helper macro, stringifies the argument
 */
#define CRGPU_STRINGIFY(X) CRGPU_STRINGIFY_HELPER(X)

/** @brief Helper macro for concatenation.
 */
#define CRGPU_STRINGIFY_HELPER(...) #__VA_ARGS__


/** @brief Helper macro, returns the first argument
 */
#define CRGPU_GET_FIRST(X, ...) X

/** @brief Helper macro, returns the second argument
 */
#define CRGPU_GET_SECOND(X, ...) CRGPU_GET_FIRST(__VA_ARGS__, X)

/** @brief Definitions to check for list end
 */
#define CRGPU_TEST_CRGPU_END ,1
/** @brief Definitions to allow an empty argument to also be used as a terminator
 */
#define CRGPU_TEST_ ,1

/** @brief Check for CRGPU_END (or an empty argument) to terminate list
 */
#define CRGPU_CHECK_FOR_END(X) CRGPU_CHECK_FOR_END_CHECK(CRGPU_TEST_ ## X, 0)

/** @brief Implementation of checking for CRGPU_END (or an empty argument) to terminate list
 */
#define CRGPU_CHECK_FOR_END_CHECK(...) CRGPU_GET_SECOND(__VA_ARGS__,)

/** @brief Expands recursive macros. Wrap the recursive macro call in `CRGPU_RECURSIVE_MACRO` to make it work.
 */
#define CRGPU_RECURSIVE_MACRO(...) CR_GPU_MACRO_RECURSE1(CR_GPU_MACRO_RECURSE1(CR_GPU_MACRO_RECURSE1(CR_GPU_MACRO_RECURSE1(__VA_ARGS__))))

/** @brief Helper macro to expand recursive macros (recursion level 1).
 */
#define CR_GPU_MACRO_RECURSE1(...) CR_GPU_MACRO_RECURSE2(CR_GPU_MACRO_RECURSE2(CR_GPU_MACRO_RECURSE2(CR_GPU_MACRO_RECURSE2(__VA_ARGS__))))

/** @brief Helper macro to expand recursive macros (recursion level 2).
 */
#define CR_GPU_MACRO_RECURSE2(...) CR_GPU_MACRO_RECURSE3(CR_GPU_MACRO_RECURSE3(CR_GPU_MACRO_RECURSE3(CR_GPU_MACRO_RECURSE3(__VA_ARGS__))))

/** @brief Helper macro to expand recursive macros (recursion level 3).
 */
#define CR_GPU_MACRO_RECURSE3(...) CR_GPU_MACRO_RECURSE4(CR_GPU_MACRO_RECURSE4(CR_GPU_MACRO_RECURSE4(CR_GPU_MACRO_RECURSE4(__VA_ARGS__))))

/** @brief Helper macro to expand recursive macros (recursion level 4).
 */
#define CR_GPU_MACRO_RECURSE4(...) CR_GPU_MACRO_RECURSE_BASE(CR_GPU_MACRO_RECURSE_BASE(CR_GPU_MACRO_RECURSE_BASE(CR_GPU_MACRO_RECURSE_BASE(__VA_ARGS__))))

/** @brief Helper macro to expand recursive macros (last recursion level: simply expands the macro).
 */
#define CR_GPU_MACRO_RECURSE_BASE(...) __VA_ARGS__

/** @brief Helper macro to apply a macro to all elements of the variadic list,
           with @p EXTRA_ARG and the parenthesised rest of the arguments
           being passed as arguments to the macro too.
 */
#define CRGPU_MACRO_EXPANSION(MACRO, EXTRA_ARG, ... )                             \
  CRGPU_MACRO_EXPANSION_IMPL(MACRO, EXTRA_ARG, __VA_ARGS__, CRGPU_END, CRGPU_END)

#define CRGPU_MACRO_EXPANSION_IMPL(MACRO, EXTRA_ARG, ONE, NEXT, ... )                                                                 \
  CRGPU_DEFER(CRGPU_CONCAT(CRGPU_MACRO_INSTANTIATION_IMPL_, CRGPU_CHECK_FOR_END(ONE))) (MACRO) (ONE, EXTRA_ARG, (NEXT, __VA_ARGS__) ) \
  CRGPU_DEFER(CRGPU_CONCAT(CRGPU_MACRO_EXPANSION_IMPL_, CRGPU_CHECK_FOR_END(NEXT))) () (MACRO, EXTRA_ARG, NEXT, __VA_ARGS__ )

/** @brief Helper macro to defer macro expansion to enable recursion.
  */
#define CRGPU_MACRO_EXPANSION_IMPL_0 CRGPU_MACRO_EXPANSION_IMPL_DO

/** @brief Helper macro to defer macro expansion to enable recursion.
  */
#define CRGPU_MACRO_EXPANSION_IMPL_DO() CRGPU_MACRO_EXPANSION_IMPL

/** @brief Helper macro to defer macro expansion to enable recursion.
  */
#define CRGPU_MACRO_EXPANSION_IMPL_1 CRGPU_MACRO_EXPANSION_IMPL_DONT

/** @brief Helper macro to defer macro expansion to enable recursion.
  */
#define CRGPU_MACRO_EXPANSION_IMPL_DONT() CRGPU_IGNORE

/** @brief Helper macro to defer macro expansion to enable recursion.
  */
#define CRGPU_MACRO_INSTANTIATION_IMPL_0 CRGPU_MACRO_INSTANTIATION_DO

/** @brief Helper macro to defer macro expansion to enable recursion.
  */
#define CRGPU_MACRO_INSTANTIATION_IMPL_1 CRGPU_MACRO_INSTANTIATION_DONT

/** @brief Helper macro to defer macro expansion to enable recursion.
  */
#define CRGPU_MACRO_INSTANTIATION_DO(...) __VA_ARGS__

/** @brief Helper macro to defer macro expansion to enable recursion.
  */
#define CRGPU_MACRO_INSTANTIATION_DONT(...) CRGPU_IGNORE

/** @brief Base macro to be applied to every element for converting strings to enums.
  */
#define CRGPU_STRING_TO_ENUM_OP(TYPE, VAR_PREFIX_PAIR, IGNORE)        \
  else if (CRGPU_GET_FIRST VAR_PREFIX_PAIR == CRGPU_STRINGIFY(TYPE))  \
    { return CRGPU_GET_SECOND VAR_PREFIX_PAIR::TYPE; }


/** @brief Checks a string variable, @p VAR,
    for matching enum identifiers (@p ONE and the remaining variadic arguments),
    for the enum starting with @p PREFIX.

    @warning Should be wrapped in `CRGPU_RECURSIVE_MACRO` to work properly.
  */
#define CRGPU_CHEAP_STRING_TO_ENUM(VAR, PREFIX, ONE, ...) \
  if (false) { return PREFIX::ONE; } CRGPU_MACRO_EXPANSION(CRGPU_STRING_TO_ENUM_OP, (VAR, PREFIX), ONE, __VA_ARGS__)


//Implements this in any C++ compiler that supports variadic macros
//(MSVC had some problems in the past, not sure how it is now...)


//To call this, wrap in CRGPU_RECURSIVE_MACRO( ... ) to ensure correct expansion.
//If there are too many items and the final macros come unexpanded,
//thus leading to "weird" compilation errors, maybe wrap it in two or three...
//Any question, reach me at nuno.dos.santos.fernandes@cern.ch


#endif //CALORECGPU_MACROHELPERS_H