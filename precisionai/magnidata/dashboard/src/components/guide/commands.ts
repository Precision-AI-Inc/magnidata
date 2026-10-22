/**
 * Every shell command shown in the guide, in one place so tests can check them against the real tools
 * (flags that exist, files that are referenced) and chapters never drift apart.
 */
export const COMMANDS = {
  clone: 'git clone git@github.com:Precision-AI-Inc/magnidata.git\ncd magnidata',
  up: 'docker compose up --build',
  ps: 'docker compose ps',
  health: 'curl http://localhost:5051/api/health',
  gpuEnv: 'echo "COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml" >> .env',
  gpuCheck: 'docker compose exec trainer python -c "import torch; print(torch.cuda.is_available())"',
  logs: 'docker compose logs -f api',
  stop: 'docker compose down',
  reset: 'docker compose down -v',
  venv: 'python -m venv .venv\nsource .venv/bin/activate\npip install -e .',
  partitionDry: 'python -m precisionai.magnidata.scripts.partition_dataset --limit-images 8 --parts 2 --dry-run',
  partition: 'python -m precisionai.magnidata.scripts.partition_dataset --limit-images 8 --parts 2',
  partitionFull: 'python -m precisionai.magnidata.scripts.partition_dataset --parts 4',
  partitionHelp: 'python -m precisionai.magnidata.scripts.partition_dataset --help',
  copyFolder: 'cp -r ~/Pictures/my-field-survey image_sets/',
  restartApi: 'docker compose restart api',
  mlInstall: 'pip install -e ".[ml]"',
  embeddings:
    'python -m precisionai.magnidata.tools.embeddings \\\n  --input precisionai/magnidata/data_user/my-field-survey.csv \\\n  --output precisionai/magnidata/data_user/my-field-survey.json \\\n  --model dinov2 --device cpu',
  embeddingsVariant:
    'python -m precisionai.magnidata.tools.embeddings \\\n  --input precisionai/magnidata/data_user/my-field-survey.csv \\\n  --output precisionai/magnidata/data_user/my-field-survey_resnet18.json \\\n  --model resnet18 --device cpu',
  trainerTests: 'python -m pytest precisionai/magnidata/training -q',
  mlflowEnv: 'export MLFLOW_TRACKING_URI=http://localhost:5052',
  agrituneClone: 'git clone git@github.com:Precision-AI-Inc/agritune.git ../agritune',
  // Extends an existing COMPOSE_FILE (e.g. the GPU file) instead of appending a second line, which would override it.
  agrituneEnable:
    "if grep -q '^COMPOSE_FILE=' .env 2>/dev/null; then\n  grep -q 'docker-compose.agritune.yml' .env || sed -i '/^COMPOSE_FILE=/s/$/:docker-compose.agritune.yml/' .env\nelse\n  echo 'COMPOSE_FILE=docker-compose.yml:docker-compose.agritune.yml' >> .env\nfi\ndocker compose up -d --build",
  agrituneEncoder: "echo 'AGRITUNE_ENCODER_API_KEY=replace-me' >> ../agritune/.env\ndocker compose up -d --build trainer",
  agrituneOffline: "echo 'AGRITUNE_ENCODER_BASE_URL=' >> .env\ndocker compose up -d --build trainer",
  features:
    'python -m precisionai.magnidata.tools.features --input images.csv --output features.csv --skip-nima',
} as const

export type CommandKey = keyof typeof COMMANDS
