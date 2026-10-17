# Checkpoints e artefatos ONNX

Este diretório não inclui pesos pretreinados. A exportação de um checkpoint
pretreinado autorizado permanece **não validada**. Os testes usam somente um
CRNN sintético local; não baixam pesos nem imagens.

## Download explícito

Instale as ferramentas de desenvolvimento e informe um SHA imutável de 40
caracteres hexadecimais minúsculos do repositório do modelo:

```sh
uv sync --extra dev --extra export --extra benchmark
export MODEL_REVISION='SUBSTITUA_PELO_SHA_DO_CHECKPOINT_AUTORIZADO'
uv run --extra export python scripts/fetch_checkpoint.py \
  --repo-id jtrecenti/txtcaptcha-crnn \
  --revision-sha "$MODEL_REVISION" \
  --destination models/checkpoint
```

Somente `model.safetensors` e `config.json` são solicitados ao Hub. Branches
como `main`, tags e SHAs abreviados são rejeitados antes do download. O destino
precisa ser novo, para não misturar arquivos de revisões diferentes. Um download
incompleto gera erro; remova seu destino incompleto antes de tentar novamente.

## Exportação offline

```sh
uv run --extra export python scripts/export_onnx.py \
  --source-dir models/checkpoint \
  --out-dir models/captix-v1-onnx-fp32 \
  --revision-sha "$MODEL_REVISION" \
  --source-commit "$(git rev-parse HEAD)"
```

O exportador usa `captix.hub.load_local()`, CPU, FP32 e modo `eval()`. Não há
acesso remoto nem carregamento de arquivos pickle. O diretório de saída precisa
ser novo; os artefatos só são publicados após a exportação e o ONNX checker
terminarem. Uma falha não publica saída parcial nem substitui artefatos antigos.
Antes de carregar os pesos, exige `architecture="CRNN"` e `vocab` explícito
com os 62 caracteres `0-9a-zA-Z`, exatamente na ordem original. Vocabulário
ausente, alterado, duplicado ou reordenado é rejeitado sem assumir um padrão.

O contrato é `images: float32[1,3,32,W]` para `logits: float32[T,1,C]`, com
`T=floor(W/4)`, `C=len(vocab)+1`, blank `0`, RGB na escala `0..1` e largura
contratada de `4..1024`. Apenas W e T são dinâmicos; o runtime deve impor a faixa
de largura antes de executar o grafo.

### Compatibilidade do exportador

Com `torch==2.11.0`, `dynamo=True` especializa a sequência da LSTM em `W=160`
e rejeita a largura dinâmica. Por decisão explícita do controlador da Task 5,
esta ferramenta usa `dynamo=False`, opset 18 e eixos dinâmicos declarados.
Não existe fallback automático nem exportação silenciosa com largura fixa.

O pooling `AdaptiveAvgPool2d((1,None))` também não é exportável por essa via
com W dinâmico. Somente na cópia carregada para exportação, ele é substituído
pela média sobre a altura com `keepdim=True`. Os arquivos do modelo de referência
e seus pesos permanecem intactos. A equivalência de pooling tem tolerância
`atol=rtol=1e-6`; logits ONNX versus PyTorch são comparados com
`atol=1e-5, rtol=1e-4` em larguras 4, 31, 160 e 1024 no CRNN sintético.
PyTorch emite avisos de depreciação e tracing por essa via; a versão está fixada.

## Manifesto

`manifest.json` tem `schema_version=1` e `model_version=captix-v1-onnx-fp32`.
Registra SHA-256 de `model.onnx`, `model.safetensors` e `config.json`; o commit
Captix e a revisão do checkpoint informados; versões de torch, ONNX, ONNX Script
e ONNX Runtime; backend/opset; vocabulário; blank; altura; nomes; dtype; escala;
layouts e faixa W. Os SHAs de proveniência são declarações do operador: forneça
a revisão correspondente aos arquivos locais e o commit correto do código.

`export` e `benchmark` são extras de desenvolvimento. ONNX não é dependência
padrão do pacote de treinamento. A publicação deste artefato para produção
ainda depende da validação com pesos autorizados e corpus congelado.

## Container Linux/amd64 offline

Prepare um artefato autorizado com o exportador acima, confira a proveniência e
copie somente `model.onnx` e `manifest.json` para a raiz de `models/`. A imagem
confere o contrato do manifesto e o SHA-256 durante o build e novamente no
startup. Sem esses arquivos, com hash incorreto ou sem token, o serviço falha;
nenhum modelo é baixado automaticamente. O manifesto registra proveniência
declarada pelo operador; o hash não autentica a origem dos pesos.

```sh
cp models/captix-v1-onnx-fp32/model.onnx models/model.onnx
cp models/captix-v1-onnx-fp32/manifest.json models/manifest.json
docker build --platform linux/amd64 -f deploy/Dockerfile -t captix:dev .
export CAPTIX_API_TOKEN='SUBSTITUA_POR_UM_SEGREDO_FORTE'
docker compose -f deploy/compose.yaml up -d
curl --fail http://127.0.0.1:8000/health/ready
docker compose -f deploy/compose.yaml down
```

O build precisa acessar PyPI e obter `python:3.12-slim`; o startup opera offline.
O runtime contém `captix_runtime`, `service` e as dependências mínimas fixadas
com hashes, sem instalar o pacote de treinamento, torch, torchvision ou ONNX
exporter. Para atualizar o lock com uv:

```sh
uv pip compile deploy/requirements-runtime.in -o deploy/requirements-runtime.txt \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --generate-hashes
```

O Compose usa UID/GID 10001, um worker, dois threads intra-op, limite de duas
CPUs e 1 GiB de RAM, filesystem read-only e tmpfs de 16 MiB. O processo é
non-root; o daemon Docker pode ser rootful ou rootless. Só a interface loopback
do host publica a porta 8000. `/metrics` fica desativado, e a imagem não inclui
token. O encerramento dá 30 segundos ao Uvicorn e 35 segundos ao Compose.

Para verificar offline sem publicar portas:

```sh
docker run -d --name captix-offline --network none --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=16m,mode=1777 --cpus 2 --memory 1g \
  --cap-drop ALL --security-opt no-new-privileges:true \
  -e CAPTIX_API_TOKEN captix:dev
docker exec captix-offline python -c \
  "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health/ready').status)"
docker stop --time 35 captix-offline
docker rm captix-offline
```

Aguarde o healthcheck ficar saudável antes do `docker exec`. A rede `none`
permite loopback dentro do container; port forwarding não é usado nesse teste.
`uv run pytest tests/test_container_smoke.py -q` faz build temporário com um
grafo ONNX **sintético** e testa health offline, UID, upload excessivo, ausência
das dependências de treinamento, SIGTERM e rejeição de artefatos adulterados ou
ausentes. O teste remove a imagem temporária e não publica artefatos de produção.
Docker indisponível gera skip explícito; skip não comprova funcionamento.
