Module Reference
=================

API documentation for the ``precisionai.dataviz`` package. The Flask API
(``precisionai.dataviz.api``) uses flat, sibling-style imports by design — matching how
``Dockerfile.api`` deploys it — so it can't be run as an installed console script; see
README.md for how to run it. The React dashboard (``precisionai/dataviz/dashboard``) is
not part of this Python package and is not documented here.

precisionai.dataviz.tools
--------------------------

The feature-extraction and embeddings toolkit — pure, importable modules.

.. automodule:: precisionai.dataviz.tools.features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.dataviz.tools.embeddings
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.dataviz.tools.embedding_models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.dataviz.tools.coco_labels
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.dataviz.tools.pixel_features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.dataviz.tools.instance_features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.dataviz.tools.quality_features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.dataviz.tools.nima
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.dataviz.tools.metadata_join
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.dataviz.tools.image_source
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.dataviz.tools.staging
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.dataviz.tools.provenance
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.dataviz.tools.schema
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.dataviz.scripts
-----------------------------

Standalone dataset-preparation scripts.

.. automodule:: precisionai.dataviz.scripts.prepare_coco128_dashboard
   :members:
   :undoc-members:
   :show-inheritance:
