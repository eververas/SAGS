if [ "$#" -eq 0 ]
then
    echo 'Available scenes: train, truck, playroom, drjohnson, amsterdam, bilbao, hollywood, pompidou, quebec, rome, bicycle, bonsai, counter, garden, kitchen, room, stump'
    exit
fi

gpu=-1

# tandt --------------------------
if [ $1 == 'train' ]
then
    scene='tandt_db/tandt/train'
    exp_name='exp_highk-5_gth-2_sth-6_NEW1_20250707_122446'
    voxel_size=0.01
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'truck' ]
then
    scene='tandt_db/tandt/truck'
    exp_name='exp_highk-5_gth-2_sth-6_NEW1_20250707_124034'
    voxel_size=0.01
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
# db ------------------------------
elif [ $1 == 'playroom' ]
then
    scene='tandt_db/db/playroom'
    exp_name=''
    voxel_size=0.005
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'drjohnson' ]
then
    scene='tandt_db/db/drjohnson'
    exp_name=''
    voxel_size=0.005
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
# bungeenerf ---------------------
elif [ $1 == 'amsterdam' ]
then
    scene='bungeenerf/amsterdam'
    exp_name=' '
    voxel_size=0
    lod=30
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --lod ${lod} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'bilbao' ]
then
    scene='bungeenerf/bilbao'
    exp_name='exp_highk-2_gth-2_NEW1_20250704_152917'
    voxel_size=0
    lod=30
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --lod ${lod} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'hollywood' ]
then
    scene='bungeenerf/hollywood'
    exp_name=' '
    voxel_size=0
    lod=30
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --lod ${lod} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'pompidou' ]
then
    scene='bungeenerf/pompidou'
    exp_name=' '
    voxel_size=0
    lod=30
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --lod ${lod} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'quebec' ]
then
    scene='bungeenerf/quebec'
    exp_name=' '
    voxel_size=0
    lod=30
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --lod ${lod} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'rome' ]
then
    scene='bungeenerf/rome'
    exp_name=' '
    voxel_size=0
    lod=30
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --lod ${lod} --eval --gpu ${gpu} --skip_train
# mipnerf360 -------------------
elif [ $1 == 'bicycle' ]
then
    scene='mipnerf360/bicycle'
    exp_name=' '
    voxel_size=0.001
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'garden' ]
then
    scene='mipnerf360/garden'
    exp_name='exp_highk-2_gth-2_sth-8_NEW1_NEWData_20250707_140307'
    voxel_size=0.001
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'stump' ]
then
    scene='mipnerf360/stump'
    exp_name='exp_highk-8_gth-1_sth-8_NEW1_NEWData_20250708_024124'
    voxel_size=0.001
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'room' ]
then
    scene='mipnerf360/room'
    exp_name='exp_highk-2_gth-08_sth-8_NEW1_NEWData_20250707_150924'
    voxel_size=0.001
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'counter' ]
then
    scene='mipnerf360/counter'
    exp_name='exp_highk-2_gth-08_sth-8_NEW1_NEWData_20250707_163521'
    voxel_size=0.001
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'kitchen' ]
then
    scene='mipnerf360/kitchen'
    exp_name='exp_highk-2_gth-08_sth-8_NEW1_NEWData_20250707_174310'
    voxel_size=0.001
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'bonsai' ]
then
    scene='mipnerf360/bonsai'
    exp_name='exp_highk-5_gth-08_sth-8_NEW1_NEWData_20250708_015833'
    voxel_size=0.001
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
else
    echo 'Not recognized scene name.'
fi