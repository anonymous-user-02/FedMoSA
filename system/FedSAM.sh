nohup python3 obed_main.py -algo FedMoSA -lr 0.0001 -m mosa -mn mosa -nc 5 -data liver_seg -pr 0 -pv 0 -t 1 -go experiment -gpu 0 > FedSAM_liver_5_mosa_2.log 2>&1 &
nohup python3 obed_main.py -algo FedMoSA -lr 0.0001 -m mosa -mn mosa -nc 5 -data liver_seg -pr 0 -pv 1 -t 2 -go experiment -gpu 0 > FedSAM_liver_5_mosa_4.log 2>&1 &
nohup python3 obed_main.py -algo FedMoSA -lr 0.0001 -m mosa -mn mosa -nc 4 -data brain_seg -pr 0 -pv 2 -t 3 -go experiment -gpu 0 > FedSAM_brain_4_mosa_5.log 2>&1 &

nohup python3 obed_main.py -algo FedMoSA -lr 0.0001 -m mosa -mn mosa -nc 5 -data liver_seg -pr 0 -pv 0 -t 1 -go experiment-local -gpu 0 > FedSAM_liver_5_mosa_2_local.log 2>&1 &
nohup python3 obed_main.py -algo FedMoSA -lr 0.0001 -m mosa -mn mosa -nc 5 -data liver_seg -pr 0 -pv 1 -t 2 -go experiment-local -gpu 0 > FedSAM_liver_5_mosa_4_local.log 2>&1 &
nohup python3 obed_main.py -algo FedMoSA -lr 0.0001 -m mosa -mn mosa -nc 5 -data liver_seg -pr 0 -pv 2 -t 3 -go experiment-local -gpu 0 > FedSAM_liver_5_mosa_5_local.log 2>&1 &

nohup python3 obed_main.py -algo FedMSA -lr 0.0001 -m msavanilla -mn msavanilla -nc 5 -data liver_seg -pr 0 -pv 0 -t 1 -go experiment -gpu 0 > FedSAM_liver_5_samvanilla_2.log 2>&1 &
nohup python3 obed_main.py -algo FedMSA -lr 0.0001 -m msavanilla -mn msavanilla -nc 5 -data liver_seg -pr 0 -pv 1 -t 2 -go experiment -gpu 0 > FedSAM_liver_5_samvanilla_4.log 2>&1 &
nohup python3 obed_main.py -algo FedMSA -lr 0.0001 -m msavanilla -mn msavanilla -nc 5 -data liver_seg -pr 0 -pv 2 -t 3 -go experiment -gpu 0 > FedSAM_liver_5_samvanilla_5.log 2>&1 &

nohup python3 obed_main.py -algo FedMSA -lr 0.0001 -m msavanilla -mn msavanilla -nc 5 -data liver_seg -pr 0 -pv 0 -t 1 -go experiment-local -gpu 0 > FedSAM_liver_5_samvanilla_2_local.log 2>&1 &
nohup python3 obed_main.py -algo FedMSA -lr 0.0001 -m msavanilla -mn msavanilla -nc 5 -data liver_seg -pr 0 -pv 1 -t 2 -go experiment-local -gpu 0 > FedSAM_liver_5_samvanilla_4_local.log 2>&1 &
nohup python3 obed_main.py -algo FedMSA -lr 0.0001 -m msavanilla -mn msavanilla -nc 5 -data liver_seg -pr 0 -pv 2 -t 3 -go experiment-local -gpu 0 > FedSAM_liver_5_samvanilla_5_local.log 2>&1 &

nohup python3 obed_main.py -algo FednnUNET -lr 0.01 -m nnunet -mn nnunet -nc 5 -data liver_seg -pr 0 -pv 0 -t 1 -go experiment -gpu 0 > FednnUNET_liver_5_nnunet_2.log 2>&1 &
nohup python3 obed_main.py -algo FednnUNET -lr 0.01 -m nnunet -mn nnunet -nc 5 -data liver_seg -pr 0 -pv 1 -t 2 -go experiment -gpu 0 > FednnUNET_liver_5_nnunet_4.log 2>&1 &
nohup python3 obed_main.py -algo FednnUNET -lr 0.01 -m nnunet -mn nnunet -nc 5 -data liver_seg -pr 0 -pv 2 -t 3 -go experiment -gpu 0 > FednnUNET_liver_5_nnunet_5.log 2>&1 &

nohup python3 obed_main.py -algo FednnUNET -lr 0.01 -m nnunet -mn nnunet -nc 5 -data liver_seg -afa 1 -pr 0 -pv 0 -t 1 -go experiment-afa -gpu 0 > FednnUNET_liver_5_nnunet_2_AFA.log 2>&1 &
nohup python3 obed_main.py -algo FednnUNET -lr 0.01 -m nnunet -mn nnunet -nc 5 -data liver_seg -afa 1 -pr 0 -pv 1 -t 2 -go experiment-afa -gpu 0 > FednnUNET_liver_5_nnunet_4_AFA.log 2>&1 &
nohup python3 obed_main.py -algo FednnUNET -lr 0.01 -m nnunet -mn nnunet -nc 5 -data liver_seg -afa 1 -pr 0 -pv 2 -t 3 -go experiment-afa -gpu 0 > FednnUNET_liver_5_nnunet_5_AFA.log 2>&1 &

nohup python3 obed_main.py -algo FednnUNET -lr 0.01 -m nnunet -mn nnunet -nc 5 -data liver_seg -afa 1 -pr 0 -pv 0 -t 1 -go experiment-local -gpu 2 > FednnUNET_liver_5_nnunet_2_local.log 2>&1 &
nohup python3 obed_main.py -algo FednnUNET -lr 0.01 -m nnunet -mn nnunet -nc 5 -data liver_seg -afa 1 -pr 0 -pv 1 -t 2 -go experiment-local -gpu 2 > FednnUNET_liver_5_nnunet_4_local.log 2>&1 &
nohup python3 obed_main.py -algo FednnUNET -lr 0.01 -m nnunet -mn nnunet -nc 5 -data liver_seg -afa 1 -pr 0 -pv 2 -t 3 -go experiment-local -gpu 2 > FednnUNET_liver_5_nnunet_5_local.log 2>&1 &
