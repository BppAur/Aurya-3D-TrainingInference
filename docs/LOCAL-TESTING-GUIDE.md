# Local Testing Guide (macOS)

**Objetivo:** Testar tudo que é possível localmente ANTES de gastar dinheiro no RunPod.

## O Que Podemos Testar Localmente

✅ **SIM - Podemos testar:**
- Docker Compose syntax
- Container builds (verificar Dockerfiles)
- Processing container (CPU-only, funciona no macOS)
- Scripts Python (syntax, imports, lógica)
- Config files (YAML/JSON parsing)
- Documentation

❌ **NÃO - Precisa GPU no RunPod:**
- Training container (precisa NVIDIA GPU)
- Inference container (precisa NVIDIA GPU)
- Sampling script (precisa PyTorch3D + GPU)

---

## Pré-Requisitos

```bash
# Verificar instalações
docker --version          # Docker Desktop 4.0+
docker compose --version  # v2.0+
python3 --version        # Python 3.10+
blender --version        # Blender 3.6+ (se instalado)

# Instalar Blender no macOS (opcional, para testar rendering localmente)
brew install --cask blender
```

---

## Teste 1: Validação de Sintaxe

### 1.1 Docker Compose
```bash
cd ~/Documents/Projects/UltraShape-Training

# Validar sintaxe do docker-compose.yml
docker compose config

# Deve mostrar a configuração merged sem erros
```

**Resultado esperado:** ✅ Configuração válida, sem erros de sintaxe

---

### 1.2 Validar Configs YAML/JSON
```bash
# Test YAML configs
python3 -c "import yaml; yaml.safe_load(open('configs/train_dit_refine.yaml'))" && echo "✅ train_dit_refine.yaml OK"
python3 -c "import yaml; yaml.safe_load(open('configs/infer_dit_refine.yaml'))" && echo "✅ infer_dit_refine.yaml OK"
python3 -c "import yaml; yaml.safe_load(open('configs/train_vae_refine.yaml'))" && echo "✅ train_vae_refine.yaml OK"

# Test JSON config
python3 -c "import json; json.load(open('configs/deepspeed_zero2.json'))" && echo "✅ deepspeed_zero2.json OK"

# Test .env.example
cat .env.example
```

**Resultado esperado:** ✅ Todos os arquivos são válidos

---

### 1.3 Validar Scripts Python
```bash
# Verificar sintaxe de todos os scripts
python3 -m py_compile scripts/*.py && echo "✅ All scripts OK"

# Verificar imports básicos
python3 -c "from pathlib import Path; import sys; sys.path.insert(0, '.'); from scripts import process_dataset" && echo "✅ Imports OK"
```

**Resultado esperado:** ✅ Sem erros de sintaxe ou imports

---

## Teste 2: Build dos Containers

### 2.1 Build Processing Container (funciona no macOS)
```bash
# Build container de processamento
docker compose build processing

# Verificar imagem criada
docker images | grep ultrashape-processing
```

**Tempo esperado:** ~10-15 minutos
**Resultado esperado:** ✅ Imagem criada com sucesso

---

### 2.2 Build Training Container (build funciona, run não)
```bash
# Build container de treinamento (apenas build, não run)
docker compose build training

# Verificar imagem criada
docker images | grep ultrashape-training
```

**Tempo esperado:** ~15-20 minutos
**Resultado esperado:** ✅ Imagem criada, mas NÃO conseguiremos rodar (precisa GPU)

---

### 2.3 Build Inference Container
```bash
# Build container de inferência
docker compose build inference

# Verificar imagem criada
docker images | grep ultrashape-inference
```

**Tempo esperado:** ~10-15 minutos
**Resultado esperado:** ✅ Imagem criada

---

## Teste 3: Processing Container (TESTE COMPLETO)

Este é o ÚNICO container que podemos testar completamente no macOS!

### 3.1 Preparar Dados de Teste
```bash
# Criar diretórios
mkdir -p data/input data/output

# Você precisa colocar alguns arquivos OBJ de teste aqui
# Opção 1: Copiar 3-5 modelos dos seus 30k
cp /path/to/your/models/*.obj data/input/  # Copie apenas 3-5 para teste

# Opção 2: Baixar modelos de teste
# (adicione links se quiser modelos de exemplo)
```

**IMPORTANTE:** Coloque apenas 3-5 modelos OBJ para teste rápido!

---

### 3.2 Testar Processamento Completo
```bash
# Rodar container de processamento com dados de teste
docker compose --profile processing run --rm processing \
  --input-dir /input \
  --output-dir /output \
  --num-workers 4 \
  --num-views 16 \
  --limit 3

# Monitorar logs em tempo real
```

**Tempo esperado:** ~5-10 minutos para 3 modelos
**O que vai acontecer:**
1. ✅ Watertight mesh processing (PyMeshLab)
2. ✅ Blender rendering (16 views RGBA)
3. ✅ Criação de data_list/train.json e val.json
4. ✅ Criação de render.json

---

### 3.3 Verificar Output
```bash
# Verificar estrutura de diretórios
tree -L 4 data/output

# Deve mostrar:
# data/output/
# ├── watertight/
# │   ├── model_001.obj
# │   ├── model_002.obj
# │   └── model_003.obj
# ├── renders/
# │   ├── model_001/
# │   │   └── model_001/
# │   │       └── rgba/
# │   │           ├── 000.png
# │   │           ├── 001.png
# │   │           ...
# │   │           └── 015.png
# ├── data_list/
# │   ├── train.json
# │   └── val.json
# └── render.json

# Verificar contagem de arquivos
ls data/output/watertight/*.obj | wc -l  # Deve mostrar 3
ls data/output/renders/*/*/rgba/*.png | wc -l  # Deve mostrar 48 (3 models × 16 views)
ls data/output/data_list/*.json | wc -l  # Deve mostrar 2

# Verificar formato render.json
cat data/output/render.json | python3 -m json.tool
# Deve ser um dicionário: {"model_001": "renders/model_001", ...}

# Verificar imagens são RGBA
python3 << 'EOF'
from PIL import Image
import sys

img = Image.open("data/output/renders/model_001/model_001/rgba/000.png")
print(f"Image mode: {img.mode}")
print(f"Image size: {img.size}")
print(f"Channels: {len(img.getbands())}")

if img.mode == "RGBA" and len(img.getbands()) == 4:
    print("✅ RGBA format correct!")
    sys.exit(0)
else:
    print("❌ Wrong format!")
    sys.exit(1)
EOF
```

**Resultado esperado:**
- ✅ 3 watertight meshes
- ✅ 48 imagens RGBA (3 models × 16 views)
- ✅ render.json no formato correto
- ✅ train.json e val.json com IDs

---

## Teste 4: Validação de Dados

### 4.1 Verificar Estrutura de Dados
```bash
# Criar script de validação
cat > validate_data.py << 'EOF'
#!/usr/bin/env python3
"""Validate processed data structure."""
import json
import sys
from pathlib import Path
from PIL import Image

def validate_structure(output_dir):
    output_dir = Path(output_dir)
    errors = []

    # Check render.json
    render_json = output_dir / "render.json"
    if not render_json.exists():
        errors.append("❌ render.json not found")
        return errors

    render_map = json.load(open(render_json))

    # Check each model
    for model_id, render_base in render_map.items():
        print(f"Checking {model_id}...")

        # Check watertight mesh
        watertight = output_dir / "watertight" / f"{model_id}.obj"
        if not watertight.exists():
            errors.append(f"❌ Watertight mesh missing: {model_id}")

        # Check renders
        rgba_dir = output_dir / render_base / model_id / "rgba"
        if not rgba_dir.exists():
            errors.append(f"❌ RGBA directory missing: {model_id}")
            continue

        # Check 16 views
        for i in range(16):
            img_path = rgba_dir / f"{i:03d}.png"
            if not img_path.exists():
                errors.append(f"❌ Missing view {i:03d}.png for {model_id}")
            else:
                # Check RGBA format
                try:
                    img = Image.open(img_path)
                    if img.mode != "RGBA":
                        errors.append(f"❌ Wrong format {img.mode} (expected RGBA): {img_path}")
                    if len(img.getbands()) != 4:
                        errors.append(f"❌ Wrong channels {len(img.getbands())} (expected 4): {img_path}")
                except Exception as e:
                    errors.append(f"❌ Error reading {img_path}: {e}")

    # Check data_list
    data_list_dir = output_dir / "data_list"
    if not (data_list_dir / "train.json").exists():
        errors.append("❌ train.json not found")
    if not (data_list_dir / "val.json").exists():
        errors.append("❌ val.json not found")

    return errors

if __name__ == "__main__":
    errors = validate_structure("data/output")

    if errors:
        print("\n❌ VALIDATION FAILED:")
        for error in errors:
            print(f"  {error}")
        sys.exit(1)
    else:
        print("\n✅ ALL VALIDATION PASSED!")
        print("Data structure is correct and ready for RunPod!")
        sys.exit(0)
EOF

chmod +x validate_data.py
python3 validate_data.py
```

**Resultado esperado:** ✅ ALL VALIDATION PASSED!

---

## Teste 5: Scripts Helper

### 5.1 Testar Scripts RunPod (Sintaxe)
```bash
# Verificar sintaxe dos scripts bash
bash -n scripts/runpod_setup.sh && echo "✅ runpod_setup.sh OK"
bash -n scripts/runpod_monitor.sh && echo "✅ runpod_monitor.sh OK"
bash -n scripts/train_deepspeed.sh && echo "✅ train_deepspeed.sh OK"
bash -n train.sh && echo "✅ train.sh OK"
```

**Resultado esperado:** ✅ Todos os scripts sem erros de sintaxe

---

### 5.2 Testar Download Script (Mock)
```bash
# Testar download script com --help
docker compose --profile training run --rm --entrypoint python3 training \
  scripts/download_pretrained.py --help

# Deve mostrar help message sem erros
```

**Resultado esperado:** ✅ Help message exibido

---

## Teste 6: Inference API (Syntax Test)

### 6.1 Testar Health Endpoint
```bash
# Tentar iniciar container (vai falhar por falta de GPU, mas testa sintaxe)
docker compose --profile inference run --rm --entrypoint python3 inference \
  -c "from scripts.api_server import app; print('✅ API imports OK')"
```

**Resultado esperado:** ✅ API imports OK

---

## Checklist de Validação Local

Antes de ir para o RunPod, certifique-se:

### Builds
- [ ] ✅ Processing container build completo
- [ ] ✅ Training container build completo
- [ ] ✅ Inference container build completo

### Configs
- [ ] ✅ docker-compose.yml válido
- [ ] ✅ train_dit_refine.yaml válido
- [ ] ✅ infer_dit_refine.yaml válido
- [ ] ✅ deepspeed_zero2.json válido

### Processing Pipeline (TESTE COMPLETO)
- [ ] ✅ 3 modelos processados com sucesso
- [ ] ✅ 3 watertight meshes criados
- [ ] ✅ 48 imagens RGBA criadas (3×16)
- [ ] ✅ Estrutura de diretórios correta: `{id}/{id}/rgba/NNN.png`
- [ ] ✅ render.json no formato correto
- [ ] ✅ data_list/train.json e val.json criados
- [ ] ✅ Script de validação passou

### Scripts
- [ ] ✅ Todos os scripts Python sem erros de sintaxe
- [ ] ✅ Todos os scripts Bash sem erros de sintaxe
- [ ] ✅ Imports funcionando

---

## Problemas Comuns

### "Blender not found"
Se o processing falhar com "blender not found":
```bash
# Instalar Blender
brew install --cask blender

# Ou usar docker sem Blender test local (vai funcionar no RunPod)
```

### "Permission denied"
```bash
# Dar permissões aos scripts
chmod +x scripts/*.sh scripts/*.py
```

### "No such file or directory: data/input"
```bash
# Criar diretórios
mkdir -p data/input data/output
```

---

## Próximos Passos

Depois de TODOS os testes locais passarem:

1. ✅ Commit final do código testado
2. ✅ Push para seu repositório Git
3. ✅ Seguir o STEP-BY-STEP-GUIDE.md Fase 2 (RunPod)
4. ✅ Processar 10 modelos no RunPod A40 para validar
5. ✅ Sampling + Training de teste (100 steps)
6. ✅ Escalar para 30k modelos no H100

---

## Comandos Rápidos de Teste

Execute tudo de uma vez:

```bash
#!/bin/bash
echo "🧪 Iniciando testes locais..."

# 1. Validar configs
echo "1. Validando configs..."
docker compose config > /dev/null && echo "✅ docker-compose.yml OK"
python3 -c "import yaml; yaml.safe_load(open('configs/train_dit_refine.yaml'))" && echo "✅ train_dit_refine.yaml OK"
python3 -c "import json; json.load(open('configs/deepspeed_zero2.json'))" && echo "✅ deepspeed_zero2.json OK"

# 2. Validar scripts
echo -e "\n2. Validando scripts..."
python3 -m py_compile scripts/*.py && echo "✅ All Python scripts OK"
bash -n scripts/*.sh && echo "✅ All Bash scripts OK"

# 3. Build containers
echo -e "\n3. Building containers..."
docker compose build processing && echo "✅ Processing container built"
# docker compose build training && echo "✅ Training container built"  # Demora muito
# docker compose build inference && echo "✅ Inference container built"  # Demora muito

echo -e "\n✅ Testes básicos concluídos!"
echo "📝 Próximo passo: Processar 3 modelos de teste"
echo "   docker compose --profile processing run processing --input-dir /input --output-dir /output --limit 3"
```

---

**Conclusão:** Teste tudo que puder localmente para detectar erros ANTES de gastar dinheiro no RunPod! 💰
