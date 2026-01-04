# Quick Start - Teste Local

## Passo 1: Rodar Testes Automatizados

```bash
cd ~/Documents/Projects/UltraShape-Training

# Rodar todos os testes (sem build de containers)
SKIP_BUILD=1 bash scripts/test_local.sh
```

**Resultado esperado:** ✅ ALL TESTS PASSED! (24/24 tests)

---

## Passo 2: Preparar Dados de Teste

```bash
# Criar diretórios
mkdir -p data/input data/output

# Copiar 3-5 modelos de teste (qualquer formato)
# Formatos suportados: OBJ, STL, FBX, PLY, OFF, 3DS, DAE, GLTF, GLB
cp /path/to/models/*.{obj,stl,fbx,ply} data/input/

# Verificar
ls -lh data/input/
```

**Nota:** O sistema detecta automaticamente o formato e converte para OBJ durante o processamento.

---

## Passo 3: Processar Dados de Teste

```bash
# Processar todos os modelos em data/input/
docker compose --profile processing run --rm processing

# Ou com limite para testes rápidos (3 modelos)
docker compose --profile processing run --rm processing --limit 3
```

**Tempo:** ~2-3 minutos por modelo (watertight processing)

**O que acontece:**
1. ✅ Detecção automática de formato (STL, OBJ, FBX, PLY, etc.)
2. ✅ Watertight mesh processing (PyMeshLab) - **Funciona no macOS**
3. ❌ Blender rendering (16 views RGBA) - **Não funciona no macOS (Rosetta)**
4. Criação de train.json e val.json
5. Criação de render.json

**Nota macOS:** Blender rendering falha no Apple Silicon. Pipeline completo requer RunPod (Linux).

---

## Passo 4: Validar Output

```bash
# Validar estrutura de dados
python3 validate_data.py --output-dir data/output
```

**Resultado esperado:** ✅ ALL VALIDATION PASSED!

**Verificações:**
- ✅ 3 watertight meshes
- ✅ 48 imagens RGBA (3 × 16 views)
- ✅ Estrutura correta: `renders/{id}/{id}/rgba/NNN.png`
- ✅ render.json com paths corretos
- ✅ data_list/train.json e val.json

---

## Passo 5: Verificar Visualmente

```bash
# Ver estrutura de arquivos
tree -L 4 data/output

# Ver uma imagem RGBA
open data/output/renders/*/*/rgba/000.png

# Verificar render.json
cat data/output/render.json | python3 -m json.tool
```

---

## Passo 6: Commit e Push

```bash
# Se tudo passou, commit
git add data/output  # Opcional: só se quiser versionar output de teste
git commit -m "test: validate local processing pipeline"
git push origin main
```

---

## Próximos Passos

✅ **Testes locais passaram?** → Siga para RunPod!

📚 **Próximo guia:** `docs/STEP-BY-STEP-GUIDE.md` Fase 2 (RunPod Setup)

**Workflow completo:**
1. ✅ Testes locais (macOS) ← VOCÊ ESTÁ AQUI
2. ⏭️ RunPod A40/A5000 (10-100 modelos de teste)
3. ⏭️ RunPod H100 (30,000 modelos produção)

---

## Troubleshooting

### Erro: "Blender not found"
```bash
# Instalar Blender
brew install --cask blender
```

### Erro: "No such file: data/input"
```bash
mkdir -p data/input data/output
# Copie seus arquivos .obj para data/input/
```

### Erro: "Permission denied"
```bash
chmod +x scripts/*.sh scripts/*.py validate_data.py
```

### Validação falhou
```bash
# Ver logs detalhados
python3 validate_data.py --output-dir data/output

# Processar novamente
rm -rf data/output
docker compose --profile processing run processing --input-dir /input --output-dir /output --limit 3
```

---

## Comandos Úteis

```bash
# Re-rodar testes completos
bash scripts/test_local.sh

# Build todos os containers (demora ~30 min)
docker compose build

# Limpar tudo e recomeçar
rm -rf data/output
docker compose down -v

# Ver logs de processamento
docker compose --profile processing run processing --help
```
