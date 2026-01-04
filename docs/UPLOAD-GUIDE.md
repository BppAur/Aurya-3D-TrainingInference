# Upload Models to RunPod

Guia para fazer upload dos seus modelos 3D do Mac para o RunPod.

---

## Método 1: Upload Direto via rsync (Recomendado)

### Passo 1: Criar Pod no RunPod

1. Acesse [RunPod Console](https://www.runpod.io/console/pods)
2. Crie um novo pod:
   - GPU: A40 ou A5000 (para testes)
   - Template: RunPod Pytorch (ou qualquer com SSH)
   - Volume: 100GB (ou mais se tiver muitos modelos)
3. Aguarde o pod iniciar

### Passo 2: Pegar Informações SSH

No RunPod Console, clique no seu pod → **Connect** → **SSH**

Você verá algo como:
```
ssh root@194.26.192.100 -p 12345 -i ~/.ssh/id_ed25519
```

Anote:
- **IP**: `194.26.192.100`
- **Porta**: `12345`

### Passo 3: Upload do Mac

No seu Mac:

```bash
cd ~/Documents/Projects/UltraShape-Training

# Sintaxe:
# bash scripts/upload_to_runpod.sh <IP> <PORTA> <CAMINHO-DOS-MODELOS>

# Exemplo:
bash scripts/upload_to_runpod.sh 194.26.192.100 12345 /Users/brunopapa/Models/Collectibles
```

**O script vai:**
1. ✅ Contar quantos modelos você tem
2. ✅ Testar conexão SSH
3. ✅ Pedir confirmação
4. ✅ Fazer upload com barra de progresso
5. ✅ Verificar que todos os arquivos subiram

### Passo 4: Verificar no RunPod

```bash
# SSH para o RunPod
ssh -p 12345 root@194.26.192.100

# Verificar modelos
ls -lh /workspace/UltraShape-Training/data/input/
```

---

## Método 2: RunPod Network Volume (Para 30k Modelos)

**Vantagens:**
- Persiste entre pods
- Não precisa re-upload
- Compartilhável

### Setup:

1. **Criar Network Volume** no RunPod Console
   - Name: `ultrashape-models`
   - Size: 500GB (ajuste conforme necessário)
   - Region: Mesma do pod

2. **Criar pod temporário** apenas para upload
   - GPU: Qualquer (mais barato)
   - Volume: Attach `ultrashape-models` em `/workspace/data`

3. **Upload inicial:**
   ```bash
   bash scripts/upload_to_runpod.sh <IP> <PORT> /path/to/models
   ```

4. **Usar em pods de produção:**
   - Sempre attach o volume `ultrashape-models`
   - Modelos estarão em `/workspace/data/input/`

---

## Método 3: Download de Cloud Storage

Se seus modelos já estão no Dropbox/Google Drive/S3:

### Dropbox

```bash
# No RunPod
cd /workspace/UltraShape-Training/data/input/

# Instalar rclone
curl https://rclone.org/install.sh | bash

# Configurar Dropbox
rclone config

# Download
rclone copy dropbox:Collectibles/Models ./ --progress
```

### Google Drive

```bash
# No RunPod
cd /workspace/UltraShape-Training/data/input/

# Instalar gdown
pip install gdown

# Se pasta pública
gdown --folder https://drive.google.com/drive/folders/FOLDER_ID

# Se pasta privada, use rclone (similar ao Dropbox)
```

### AWS S3

```bash
# No RunPod
cd /workspace/UltraShape-Training/data/input/

# Configurar AWS CLI
aws configure

# Download
aws s3 sync s3://your-bucket/models/ ./ --no-progress
```

---

## Estimativas de Tempo/Custo

### Upload via rsync (100Mbps)

| Modelos | Tamanho | Tempo | Custo Pod* |
|---------|---------|-------|------------|
| 100     | ~1GB    | ~2min | $0.02      |
| 1,000   | ~10GB   | ~20min| $0.20      |
| 10,000  | ~100GB  | ~3hr  | $2.00      |
| 30,000  | ~300GB  | ~9hr  | $6.00      |

*A40 @ $0.70/hr

**Dica:** Use Network Volume para não pagar upload toda vez!

### Network Volume Storage

| Tamanho | Custo/mês |
|---------|-----------|
| 100GB   | ~$10      |
| 300GB   | ~$30      |
| 500GB   | ~$50      |

---

## Troubleshooting

### "Connection refused"
- Pod não iniciou completamente. Aguarde 1-2 minutos.
- SSH pode não estar habilitado. Use template RunPod Pytorch.

### Upload lento
- Sua internet upload é o gargalo
- Considere usar cloud storage intermediário
- Ou deixe uploadando overnight

### "Permission denied"
- Certifique-se que está usando `root@` (não outro usuário)
- RunPod usa autenticação por chave SSH automática

### Modelos não aparecem
- Verifique se subiu para o diretório correto:
  ```bash
  ssh -p PORT root@IP "ls -lah /workspace/UltraShape-Training/data/input/"
  ```

---

## Próximos Passos

Depois do upload:

1. **SSH para o RunPod**
2. **Clone o repo:**
   ```bash
   cd /workspace
   git clone https://github.com/BppAur/Aurya-3D-TrainingInference.git UltraShape-Training
   cd UltraShape-Training
   ```

3. **Run setup:**
   ```bash
   bash scripts/runpod_setup.sh
   ```

4. **Process models:**
   ```bash
   docker compose --profile processing run processing
   ```

Veja `docs/STEP-BY-STEP-GUIDE.md` para workflow completo!
