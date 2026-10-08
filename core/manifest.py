"""Our block manifest: the blocks ``core/run.py`` walks, in order, with their config sections.

Stdlib-only on purpose: ``genia.core.utils.config`` reads it every time it resolves a per-block
output flag, and should not pay for importing the block runners. Every block listed here needs
an arm in ``Pipeline.run_block`` (``core/pipeline.py``).
"""

#: ``(block name, config section)`` in run order. The block name names the cache file
#: (``cache/{block}.pt``), the ``NN_{block}/`` output folder and the timing row; the section
#: is where the block's ``enabled`` / ``exit_after`` / ``save_*`` flags live.
BLOCKS = (
    ("gt_shapes_inversion", "gt_shapes_inversion"),
    ("shape_init", "shape_init"),
    ("appearance_init", "appearance_init"),
    ("pose_init", "pose_init"),
    ("global_pose_refine_1", "global_pose_refine_1"),
    ("appearance_init_2", "appearance_init_2"),
    ("finetune", "finetuning"),
    ("global_pose_refine_2", "global_pose_refine_2"),
)
