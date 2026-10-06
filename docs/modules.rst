Module Reference
=================

API documentation for the ``precisionai.magnidata`` package. The Flask API
(``precisionai.magnidata.api``) uses flat, sibling-style imports by design — matching how
``Dockerfile.api`` deploys it — so it can't be run as an installed console script; see
README.md for how to run it. The React dashboard (``precisionai/magnidata/dashboard``) is
not part of this Python package and is not documented here.

precisionai.magnidata.tools
--------------------------

The feature-extraction and embeddings toolkit — pure, importable modules.

.. automodule:: precisionai.magnidata.tools.features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.magnidata.tools.embeddings
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.magnidata.tools.embedding_models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.magnidata.tools.coco_labels
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.magnidata.tools.pixel_features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.magnidata.tools.instance_features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.magnidata.tools.quality_features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.magnidata.tools.nima
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.magnidata.tools.metadata_join
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.magnidata.tools.image_source
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.magnidata.tools.staging
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.magnidata.tools.provenance
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.magnidata.tools.schema
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.magnidata.scripts
-----------------------------

Standalone dataset-preparation scripts.

.. automodule:: precisionai.magnidata.scripts.prepare_coco128_dashboard
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.magnidata.training
------------------------------

The training feedback loop: the MLflow contract shared by every participant, the per-sample
loss recorder, and the reference trainer and worker. ``contract``, ``recorder`` and
``attach`` need only numpy; the others need torch.

.. automodule:: precisionai.magnidata.training.contract
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.magnidata.training.recorder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.magnidata.training.mlflow_sink
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.magnidata.training.attach
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.magnidata.training.losses
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.magnidata.training.data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.magnidata.training.train
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.magnidata.training.worker
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.magnidata.api
-------------------------

Two API modules are importable as part of the package (the rest of the API uses flat imports,
see above).

.. automodule:: precisionai.magnidata.api.mlflow_store
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.magnidata.api.run_axes
   :members:
   :undoc-members:
   :show-inheritance:
