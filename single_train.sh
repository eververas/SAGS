
if [ "$#" -eq 0 ]
then
    echo 'Available scenes: train, truck, playroom, drjohnson, amsterdam, bilbao, hollywood, pompidou, quebec, rome, bicycle, bonsai, counter, garden, kitchen, room, stump'
    exit
fi

# message for experiment output directory 
message=$(date +"%Y%m%d_%H%M%S")
if [ $2 != '' ]
then
    message="$2_$message"  
fi
echo $message

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

gpu=-1
ulimit -n 4096

# tandt --------------------------
if [ $1 == 'train' ]
then
    scene='tandt_db/tandt/train'
    exp_name="exp_${message}"
    voxel_size=0.01
    update_init_factor=16
    low_curv_ups_factor=10
    high_curv_ups_factor=5
    densify_grad_threshold=0.0002
    success_threshold=0.6
    python train.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} \
                    --upsampling_factors ${low_curv_ups_factor} ${high_curv_ups_factor} --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold}
elif [ $1 == 'truck' ]
then
    scene='tandt_db/tandt/truck'
    exp_name="exp_${message}"
    voxel_size=0.01
    update_init_factor=16
    low_curv_ups_factor=10
    high_curv_ups_factor=5
    densify_grad_threshold=0.0002
    success_threshold=0.6
    python train.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} \
                    --upsampling_factors ${low_curv_ups_factor} ${high_curv_ups_factor} --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold}
# db ------------------------------
elif [ $1 == 'playroom' ]
then
    scene='tandt_db/db/playroom'
    exp_name="exp_${message}"
    voxel_size=0.005
    update_init_factor=16
    low_curv_ups_factor=10
    high_curv_ups_factor=5
    densify_grad_threshold=0.0001
    success_threshold=0.6
    python train.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} \
                    --upsampling_factors ${low_curv_ups_factor} ${high_curv_ups_factor} --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold}
elif [ $1 == 'drjohnson' ]
then
    scene='tandt_db/db/drjohnson'
    exp_name="exp_${message}"
    voxel_size=0.005
    update_init_factor=16
    low_curv_ups_factor=10
    high_curv_ups_factor=5
    densify_grad_threshold=0.00008
    success_threshold=0.6
    python train.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} \
                    --upsampling_factors ${low_curv_ups_factor} ${high_curv_ups_factor} --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold}
# bungeenerf ---------------------
elif [ $1 == 'amsterdam' ]
then
    scene='bungeenerf/amsterdam'
    exp_name="exp_${message}"
    voxel_size=0
    update_init_factor=128
    lod=30
    low_curv_ups_factor=10
    high_curv_ups_factor=2
    densify_grad_threshold=0.0001
    success_threshold=0.8
    python train.py -s data/${scene} -m outputs/${scene}/${exp_name} --lod ${lod} --eval --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} \
                    --upsampling_factors ${low_curv_ups_factor} ${high_curv_ups_factor} --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold} --scaling_loss
elif [ $1 == 'bilbao' ]
then
    scene='bungeenerf/bilbao'
    exp_name="exp_${message}"
    voxel_size=0
    update_init_factor=128
    lod=30
    low_curv_ups_factor=10
    high_curv_ups_factor=2
    densify_grad_threshold=0.0001
    success_threshold=0.8
    python train.py -s data/${scene} -m outputs/${scene}/${exp_name} --lod ${lod} --eval --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} \
                    --upsampling_factors ${low_curv_ups_factor} ${high_curv_ups_factor} --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold} --scaling_loss

elif [ $1 == 'hollywood' ]
then
    scene='bungeenerf/hollywood'
    exp_name="exp_${message}"
    voxel_size=0
    update_init_factor=128
    lod=30
    low_curv_ups_factor=10
    high_curv_ups_factor=2
    densify_grad_threshold=0.0001
    success_threshold=0.8
    python train.py -s data/${scene} -m outputs/${scene}/${exp_name} --lod ${lod} --eval --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} \
                    --upsampling_factors ${low_curv_ups_factor} ${high_curv_ups_factor} --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold} --scaling_loss
elif [ $1 == 'pompidou' ]
then
    scene='bungeenerf/pompidou'
    exp_name="exp_${message}"
    voxel_size=0
    update_init_factor=128
    lod=30
    low_curv_ups_factor=10
    high_curv_ups_factor=2
    densify_grad_threshold=0.0001
    success_threshold=0.8
    python train.py -s data/${scene} -m outputs/${scene}/${exp_name} --lod ${lod} --eval --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} \
                    --upsampling_factors ${low_curv_ups_factor} ${high_curv_ups_factor} --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold} --scaling_loss
elif [ $1 == 'quebec' ]
then
    scene='bungeenerf/quebec'
    exp_name="exp_${message}"
    voxel_size=0
    update_init_factor=128
    lod=30
    low_curv_ups_factor=10
    high_curv_ups_factor=2
    densify_grad_threshold=0.0001
    success_threshold=0.8
    python train.py -s data/${scene} -m outputs/${scene}/${exp_name} --lod ${lod} --eval --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} \
                    --upsampling_factors ${low_curv_ups_factor} ${high_curv_ups_factor} --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold} --scaling_loss
elif [ $1 == 'rome' ]
then
    scene='bungeenerf/rome'
    exp_name="exp_${message}"
    voxel_size=0
    update_init_factor=128
    lod=30
    low_curv_ups_factor=10
    high_curv_ups_factor=2
    densify_grad_threshold=0.0001
    success_threshold=0.8
    python train.py -s data/${scene} -m outputs/${scene}/${exp_name} --lod ${lod} --eval --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} \
                    --upsampling_factors ${low_curv_ups_factor} ${high_curv_ups_factor} --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold} --scaling_loss
# mipnerf360 -------------------
elif [ $1 == 'bicycle' ]
then
    scene='mipnerf360/bicycle'
    exp_name="exp_${message}"
    voxel_size=0.001
    update_init_factor=16
    low_curv_ups_factor=10
    high_curv_ups_factor=8
    densify_grad_threshold=0.0002
    success_threshold=0.6
    python train.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} \
                    --upsampling_factors ${low_curv_ups_factor} ${high_curv_ups_factor} --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold}
elif [ $1 == 'garden' ]
then
    scene='mipnerf360/garden'
    exp_name="exp_${message}"
    voxel_size=0.001
    update_init_factor=16
    low_curv_ups_factor=10
    high_curv_ups_factor=2
    densify_grad_threshold=0.0002
    success_threshold=0.8
    python train.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} \
                    --upsampling_factors ${low_curv_ups_factor} ${high_curv_ups_factor} --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold}
elif [ $1 == 'stump' ]
then
    scene='mipnerf360/stump'
    exp_name="exp_${message}"
    voxel_size=0.001
    update_init_factor=16
    low_curv_ups_factor=10
    high_curv_ups_factor=8
    densify_grad_threshold=0.0001
    success_threshold=0.8
    python train.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} \
                    --upsampling_factors ${low_curv_ups_factor} ${high_curv_ups_factor} --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold}
elif [ $1 == 'room' ]
then
    scene='mipnerf360/room'
    exp_name="exp_${message}"
    voxel_size=0.001
    update_init_factor=16
    low_curv_ups_factor=10
    high_curv_ups_factor=2
    densify_grad_threshold=0.00008
    success_threshold=0.8
    python train.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} \
                    --upsampling_factors ${low_curv_ups_factor} ${high_curv_ups_factor} --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold}
elif [ $1 == 'counter' ]
then
    scene='mipnerf360/counter'
    exp_name="exp_${message}"
    voxel_size=0.001
    update_init_factor=16
    low_curv_ups_factor=10
    high_curv_ups_factor=2
    densify_grad_threshold=0.00008
    success_threshold=0.8
    python train.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} \
                    --upsampling_factors ${low_curv_ups_factor} ${high_curv_ups_factor} --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold}
elif [ $1 == 'kitchen' ]
then
    scene='mipnerf360/kitchen'
    exp_name="exp_${message}"
    voxel_size=0.001
    update_init_factor=16
    low_curv_ups_factor=10
    high_curv_ups_factor=2
    densify_grad_threshold=0.00008
    success_threshold=0.8
    python train.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} \
                    --upsampling_factors ${low_curv_ups_factor} ${high_curv_ups_factor} --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold}
elif [ $1 == 'bonsai' ]
then
    scene='mipnerf360/bonsai'
    exp_name="exp_${message}"
    voxel_size=0.001
    update_init_factor=16
    low_curv_ups_factor=10
    high_curv_ups_factor=5
    densify_grad_threshold=0.00008
    success_threshold=0.8
    python train.py -s data/${scene} -m outputs/${scene}/${exp_name} --eval --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} \
                    --upsampling_factors ${low_curv_ups_factor} ${high_curv_ups_factor} --densify_grad_threshold ${densify_grad_threshold} --success_threshold ${success_threshold}
else
    echo 'Not recognized scene name.'
fi