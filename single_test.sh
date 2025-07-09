if [ "$#" -eq 0 ]
then
    echo 'Available scenes: train, truck, playroom, drjohnson, amsterdam, bilbao, hollywood, pompidou, quebec, rome, bicycle, bonsai, counter, garden, kitchen, room, stump'
    exit
fi

exp_name=$2
gpu=-1

# tandt --------------------------
if [ $1 == 'train' ]
then
    scene='tandt_db/tandt/train'
    voxel_size=0.01
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'truck' ]
then
    scene='tandt_db/tandt/truck'
    voxel_size=0.01
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
# db ------------------------------
elif [ $1 == 'playroom' ]
then
    scene='tandt_db/db/playroom'
    voxel_size=0.005
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'drjohnson' ]
then
    scene='tandt_db/db/drjohnson'
    voxel_size=0.005
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
# bungeenerf ---------------------
elif [ $1 == 'amsterdam' ]
then
    scene='bungeenerf/amsterdam'
    voxel_size=0
    lod=30
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --lod ${lod} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'bilbao' ]
then
    scene='bungeenerf/bilbao'
    voxel_size=0
    lod=30
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --lod ${lod} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'hollywood' ]
then
    scene='bungeenerf/hollywood'
    voxel_size=0
    lod=30
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --lod ${lod} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'pompidou' ]
then
    scene='bungeenerf/pompidou'
    voxel_size=0
    lod=30
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --lod ${lod} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'quebec' ]
then
    scene='bungeenerf/quebec'
    voxel_size=0
    lod=30
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --lod ${lod} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'rome' ]
then
    scene='bungeenerf/rome'
    voxel_size=0
    lod=30
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --lod ${lod} --eval --gpu ${gpu} --skip_train
# mipnerf360 -------------------
elif [ $1 == 'bicycle' ]
then
    scene='mipnerf360/bicycle'
    voxel_size=0.001
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'garden' ]
then
    scene='mipnerf360/garden'
    voxel_size=0.001
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'stump' ]
then
    scene='mipnerf360/stump'
    voxel_size=0.001
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'room' ]
then
    scene='mipnerf360/room'
    voxel_size=0.001
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'counter' ]
then
    scene='mipnerf360/counter'
    voxel_size=0.001
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'kitchen' ]
then
    scene='mipnerf360/kitchen'
    voxel_size=0.001
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
elif [ $1 == 'bonsai' ]
then
    scene='mipnerf360/bonsai'
    voxel_size=0.001
    python render_and_evaluate.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --skip_train
else
    echo 'Not recognized scene name.'
fi
