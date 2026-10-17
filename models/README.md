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
