# """============= Baseline Runs --- biblept -- Using All Tokens ===================="""
python my_main.py --log_online --arch distilbert_all --token_max_length 95  --bs 64 --n_epochs 50 --lr 0.00001 --seed 0 --loss margin --dataset biblept --batch_mining distance --project DML_Project_biblept --group Margin_with_Distance_All --gpu 1 --data_sampler class_random --samples_per_class 2 --source /content/drive/MyDrive/Facul/TCC/CodigoArtigo/datasets --evaluate_on_gpu

python my_main.py --log_online --arch distilbert_all --token_max_length 75  --bs 64 --n_epochs 50 --lr 0.00001 --seed 0 --loss margin --dataset biblept --batch_mining distance --project DML_Project_biblept --group Margin_with_Distance_All --gpu 1 --data_sampler class_random --samples_per_class 2 --source /content/drive/MyDrive/Facul/TCC/CodigoArtigo/datasets --evaluate_on_gpu

python my_main.py --log_online --arch distilbert_all --token_max_length 55  --bs 64 --n_epochs 50 --lr 0.00001 --seed 0 --loss margin --dataset biblept --batch_mining distance --project DML_Project_biblept --group Margin_with_Distance_All --gpu 1 --data_sampler class_random --samples_per_class 2 --source /content/drive/MyDrive/Facul/TCC/CodigoArtigo/datasets --evaluate_on_gpu

python my_main.py --log_online --arch distilbert_all --token_max_length 35  --bs 64 --n_epochs 50 --lr 0.00001 --seed 0 --loss margin --dataset biblept --batch_mining distance --project DML_Project_biblept --group Margin_with_Distance_All --gpu 1 --data_sampler class_random --samples_per_class 2 --source /content/drive/MyDrive/Facul/TCC/CodigoArtigo/datasets --evaluate_on_gpu

python my_main.py --log_online --arch distilbert_all --token_max_length 15  --bs 64 --n_epochs 50 --lr 0.00001 --seed 0 --loss margin --dataset biblept --batch_mining distance --project DML_Project_biblept --group Margin_with_Distance_All --gpu 1 --data_sampler class_random --samples_per_class 2 --source /content/drive/MyDrive/Facul/TCC/CodigoArtigo/datasets --evaluate_on_gpu


# """============= Baseline Runs --- biblept -- Using First Token ===================="""
python my_main.py --log_online --arch distilbert_first --token_max_length 95  --bs 64 --n_epochs 50 --lr 0.00001 --seed 0 --loss margin --dataset biblept --batch_mining distance --project DML_Project_biblept --group Margin_with_Distance_First --gpu 1 --data_sampler class_random --samples_per_class 2 --source /content/drive/MyDrive/Facul/TCC/CodigoArtigo/datasets --evaluate_on_gpu

python my_main.py --log_online --arch distilbert_first --token_max_length 75  --bs 64 --n_epochs 50 --lr 0.00001 --seed 0 --loss margin --dataset biblept --batch_mining distance --project DML_Project_biblept --group Margin_with_Distance_First --gpu 1 --data_sampler class_random --samples_per_class 2 --source /content/drive/MyDrive/Facul/TCC/CodigoArtigo/datasets --evaluate_on_gpu

python my_main.py --log_online --arch distilbert_first --token_max_length 55  --bs 64 --n_epochs 50 --lr 0.00001 --seed 0 --loss margin --dataset biblept --batch_mining distance --project DML_Project_biblept --group Margin_with_Distance_First --gpu 1 --data_sampler class_random --samples_per_class 2 --source /content/drive/MyDrive/Facul/TCC/CodigoArtigo/datasets --evaluate_on_gpu

python my_main.py --log_online --arch distilbert_first --token_max_length 35  --bs 64 --n_epochs 50 --lr 0.00001 --seed 0 --loss margin --dataset biblept --batch_mining distance --project DML_Project_biblept --group Margin_with_Distance_First --gpu 1 --data_sampler class_random --samples_per_class 2 --source /content/drive/MyDrive/Facul/TCC/CodigoArtigo/datasets --evaluate_on_gpu

python my_main.py --log_online --arch distilbert_first --token_max_length 15  --bs 64 --n_epochs 50 --lr 0.00001 --seed 0 --loss margin --dataset biblept --batch_mining distance --project DML_Project_biblept --group Margin_with_Distance_First --gpu 1 --data_sampler class_random --samples_per_class 2 --source /content/drive/MyDrive/Facul/TCC/CodigoArtigo/datasets --evaluate_on_gpu
