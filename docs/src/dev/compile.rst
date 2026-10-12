.. _compile_passes:

Compiler Passes
===============

The compiler uses the existing ``array`` and ``Primitive`` graph. Its internal
pass interface is in ``mlx/compile_impl.h``. A pass takes a ``CompileGraph&``
and a ``const CompileContext&`` and returns ``void``.

Adding a Pass
-------------

Implement a function with that signature and add a ``PassSpec`` to a static
stage list in ``mlx/compile_passes.cpp``. The name identifies the pass in logs.
Use ``CompileAndExport`` for graph rewrites that preserve export behavior,
or ``CompileOnly`` for backend fusion. A pass can contain several related
rules. Additional rule families can live in separate source files and use
the same registration. Tests can call ``run_compile_passes`` with a small
list directly.

Use the graph's tape as a candidate worklist. Recheck graph membership,
consumers, and legality before each edit. Call ``replace_outputs`` to replace
ordered values together, including multi-output primitives and repeated roots.
Do not keep references to graph analyses across edits. The runner refreshes
the tape after each pass; validation builds also check graph invariants.
The runner does not repeat the stage list. A local worklist or loop must have
a termination argument.

Use ``context.same_shape`` and ``context.dimension_is`` for existing shape
queries. Only ``True`` proves a condition. In shapeless mode, sample sizes
do not become fixed dimensions. Rank, dtype, primitive attributes, and other
ordinary conditions can be checked directly. Replacement construction runs
under the same tracing and export context as the pass.

Diagnostics
-----------

``context.log`` is shared by all rules and supports all five
``CompileLogCategory`` values. The runner supplies the compilation ID, stage,
and pass name. Add a rule name, candidate node, check stage, and reason to a
candidate event. After a successful edit, use ``context.log_rewrite`` for each
replaced output. Shape queries log their proof result through the same scope.

``CompileLog::write`` takes a message function and calls it only when its
category is enabled. Keep formatting inside that function:

.. code-block:: c++

   context.log.write(CompileLogCategory::Guards, [&] {
     return fmt::format(
         "rule=simplify_views check_stage=compile node={} reason=shape_not_proven",
         candidate.id());
   });

Do not read tensor data or evaluate arrays for diagnostics. Guard graph
snapshots or other work outside the message function with ``log.enabled``.
Backend events use the original compilation ID and the ``Compiled`` primitive
ID; these fields do not participate in kernel equivalence or cache keys.

Extending the Stages
--------------------

The current stages are ``canonicalize`` and ``fusion``. Keep registration
explicit as more rules are added. The `compile RFC
<https://github.com/ml-explore/mlx/issues/4681>`_ identifies these extension
points:

* Canonicalization: identities, uniform values, dtype conversions, rsqrt,
  views, axis permutations, and split/concatenation patterns.
* Recognition before fusion: attention, convolution-to-matmul before AddMM,
  reduction reuse, uniform cumulative sums, restricted scatter, and sampling.
* Fusion: horizontal pointwise regions, view indexing, concatenation movement
  with an accepted fusion, GEMM epilogues, and reductions.

Add a recognition stage when its first pass is implemented. Keep numerical
conditions within each rule. Introduce shared value, index, or domain analyses
when a rule needs them, with invalidation handled by ``CompileGraph``. Equal
shapes alone do not prove equal element mappings. Extended fused regions must
define their backend semantics and output shape inference before matching is
enabled. The present interface does not implement dynamic-axis annotations or
those additional analyses.
