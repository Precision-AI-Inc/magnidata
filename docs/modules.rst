Module Reference
=================

API documentation for the ``precisionai.agriviz`` package. The Flask API
(``precisionai.agriviz.api``) uses flat, sibling-style imports by design — matching how
``Dockerfile.api`` deploys it — so it can't be run as an installed console script; see
README.md for how to run it. The React dashboard (``precisionai/agriviz/dashboard``) is
not part of this Python package and is not documented here.

precisionai.agriviz.tools
--------------------------

The feature-extraction and embeddings toolkit — pure, importable modules.

.. automodule:: precisionai.agriviz.tools.features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.agriviz.tools.embeddings
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.agriviz.tools.embedding_models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.agriviz.tools.coco_labels
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.agriviz.tools.pixel_features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.agriviz.tools.instance_features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.agriviz.tools.quality_features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.agriviz.tools.nima
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.agriviz.tools.metadata_join
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.agriviz.tools.image_source
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.agriviz.tools.staging
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.agriviz.tools.provenance
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: precisionai.agriviz.tools.schema
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agriviz.scripts
-----------------------------

Standalone dataset-preparation scripts.

.. automodule:: precisionai.agriviz.scripts.prepare_coco128_dashboard
   :members:
   :undoc-members:
   :show-inheritance:
