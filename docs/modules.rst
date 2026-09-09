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
