file(REMOVE_RECURSE
  "libes_hccl.pdb"
  "libes_hccl.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/es_hccl_so.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
