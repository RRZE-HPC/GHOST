#GHOST code duplication (unrolling)
BEGIN {FS="#"}
{
    if($2=="GHOST_UNROLL") {
        N=$4; 
        for (i=0; i<N; i++) {
            LINE=$3; 
            gsub(/@/,i,LINE); 
            print $1 LINE;
        }
    } else {
        print
    }
} 
