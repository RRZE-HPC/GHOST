#GHOST function variant expansion
BEGIN {FS="#"}
{
    if ($2=="GHOST_FUNC_BEGIN") {
        split($3,v,","); # store variants to generate in array v
        f=1; # function body begins
        t="";
        next;
    } else if (f==1 && $2!="GHOST_FUNC_END") {
        if (length(t)) { # append newline if t is non-empty
            t = t"\n";
        }
        t=t$0; # append line
    } else if ($2=="GHOST_FUNC_END") {
        f=0;
        for (i=1; i<length(v)+1; i++) {
            x=t;
            gsub(/\$/,v[i],x);
            print x"\n";
        }
    } else {
        print;
    }
}
