import zipfile
import json

MODEL_PATH = r"C:\Users\Vinny\Project\Python\Deploy\model\tbc_model.keras"
CLEAN_MODEL_PATH = r"C:\Users\Vinny\Project\Python\Deploy\backend\model_clean.keras"

print("Membersihkan model...")
with zipfile.ZipFile(MODEL_PATH, 'r') as zin:
    with zipfile.ZipFile(CLEAN_MODEL_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == 'config.json':
                config_str = zin.read(item.filename).decode('utf-8')
                config_data = json.loads(config_str)
                
                def remove_invalid_keys(data):
                    if isinstance(data, dict):
                        data.pop("quantization_config", None)
                        for value in data.values():
                            remove_invalid_keys(value)
                    elif isinstance(data, list):
                        for item in data:
                            remove_invalid_keys(item)
                            
                remove_invalid_keys(config_data)
                zout.writestr(item, json.dumps(config_data).encode('utf-8'))
            else:
                zout.writestr(item, zin.read(item.filename))
print(f"Model tersimpan di: {CLEAN_MODEL_PATH}")