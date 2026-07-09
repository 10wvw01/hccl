file(REMOVE_RECURSE
  "libes_hccl.a"
  "libes_hccl.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/es_hccl_a.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
