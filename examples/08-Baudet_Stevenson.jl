import MPI

include("08-Baudet_Stevenson-impl.jl")

function main()
  # Initialize MPI
  MPI.Init()

  # Create the data and locally sort
  N = 10
  x = sort(rand(N))

  # Do the parallel sort
  comm = MPI.COMM_WORLD
  psort!(x, comm)

  # Write the result to a file
  fid = open(@sprintf("output/psort_%03d.txt", MPI.Comm_rank(comm)), "w")
  for k = 1:length(x)
    println(fid, x[k])
  end
  close(fid)

  # Cleanup
  MPI.Finalize()
end

main()
