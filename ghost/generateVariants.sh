#!/bin/bash


for folder in `ls plugins`; do \
		cp -r plugins/$folder plugins/c_$folder; \
		cp -r plugins/$folder plugins/d_$folder; \
		cp -r plugins/$folder plugins/s_$folder; \
		cp -r plugins/$folder plugins/z_$folder; \
		find plugins/c_$folder/ -type f -exec sed -i 's/ghost_mdat_t/complex float/g;s/DATATYPECHAR/c/g' {} \; -exec bash -c "mv {} \`echo {} | sed -e 's/\(.*\)\//\1\/c_/'\`" \;; \
		find plugins/d_$folder/ -type f -exec sed -i 's/ghost_mdat_t/double/g;s/DATATYPECHAR/d/g' {} \; -exec bash -c "mv {} \`echo {} | sed -e 's/\(.*\)\//\1\/d_/'\`" \;; \
		find plugins/s_$folder/ -type f -exec sed -i 's/ghost_mdat_t/float/g;s/DATATYPECHAR/s/g' {} \; -exec bash -c "mv {} \`echo {} | sed -e 's/\(.*\)\//\1\/s_/'\`" \;; \
		find plugins/z_$folder/ -type f -exec sed -i 's/ghost_mdat_t/complex double/g;s/DATATYPECHAR/z/g' {} \; -exec bash -c "mv {} \`echo {} | sed -e 's/\(.*\)\//\1\/z_/'\`" \;; \
done
