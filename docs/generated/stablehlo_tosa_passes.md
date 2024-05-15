<!-- Autogenerated by mlir-tblgen; don't manually edit -->
### `-stablehlo-legalize-to-tosa`

_Legalize StableHLO to TOSA_

### `-stablehlo-prepare-for-tosa`

_Prepare StableHLO for legalization to TOSA_

This pass adds rewriters to make StableHLO ops more compatible with TOSA ops.
Currently simplifies stablehlo.dot_general into stablehlo.dot for easier lowering.