#!/usr/bin/env python3
import os
import glob
import subprocess
import sys

DEFAULT_EPOCHS = 3

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    agent_files = sorted(glob.glob("agent_*.jsonl"))

    if not agent_files:
        print("Aucun fichier agent_*.jsonl trouvé dans ce dossier.")
        return

    print(f"🚀 {len(agent_files)} agents trouvés :")
    for f in agent_files:
        print("   -", f)

    for data_path in agent_files:
        agent_name = os.path.splitext(os.path.basename(data_path))[0]
        output_dir = os.path.join("models", f"phi2_lora_{agent_name}")

        cmd = [
            sys.executable, "train_expert_lora.py",
            "--data_path", data_path,
            "--output_dir", output_dir,
            "--epochs", str(DEFAULT_EPOCHS),
        ]

        print("\n========================================")
        print(f"🎯 Entraînement de {agent_name}")
        print("   Dataset  :", data_path)
        print("   Modèle   :", output_dir)
        print("   Commande :", " ".join(cmd))
        print("========================================\n")

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"⚠️ Entraînement interrompu : erreur sur {agent_name}")
            break
        else:
            print(f"✅ Terminé pour {agent_name}")

    print("\n✨ Tous les entraînements terminés (ou arrêtés sur erreur).")

if __name__ == "__main__":
    main()
