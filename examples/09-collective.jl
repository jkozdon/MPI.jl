import MPI

include("collective-algorithms.jl")

# broadcast example
function bcast(root, comm)

  me = MPI.Comm_rank(comm)
  sz = MPI.Comm_size(comm)

  fid = open(@sprintf("output/collective_bcast_%03d.txt", me), "w")

  println(fid,"--------------------------------------")
  @printf(fid, "Broadcast: root = %03d\n", root)

  x = [(me, rand())]
  if me == root
    @printf(fid, "Root:     (%03d, %.8e) (initial)\n\n", x[1][1], x[1][2])
  end

  mstbcast!(x, root, comm)

  if me == root
    @printf(fid, "Root:     (%03d, %.8e) (final)\n", x[1][1], x[1][2])
  else
    @printf(fid, "Non-Root: (%03d, %.8e) (final)\n", x[1][1], x[1][2])
  end
  println(fid,"--------------------------------------")
  close(fid)
end

# reduce example
function reduce(root, comm)

  me = MPI.Comm_rank(comm)
  sz = MPI.Comm_size(comm)

  fid = open(@sprintf("output/collective_reduce_%03d.txt", me), "w")

  println(fid,"--------------------------------------")
  @printf(fid, "Reduce: root = %03d\n", root)

  x = [(me, rand())]
  if me == root
    @printf(fid, "Root:     (%03d, %.8e) (initial)\n\n", x[1][1], x[1][2])
  else
    @printf(fid, "Non-Root: (%03d, %.8e (initial))\n", x[1][1], x[1][2])
  end

  add(a,b) = (a[1][1] + b[1][1], a[1][2] + b[1][2])
  mstreduce!(x, add, root, comm)

  if me == root
    @printf(fid, "Root:     (%03d, %.8e) (final)\n", x[1][1], x[1][2])
  end
  println(fid,"--------------------------------------")
  close(fid)
end

# scatter example
function scatter(root, comm)

  me = MPI.Comm_rank(comm)
  sz = MPI.Comm_size(comm)

  fid = open(@sprintf("output/collective_scatter_%03d.txt", me), "w")

  println(fid,"--------------------------------------")
  @printf(fid, "Scatter: root = %03d\n", root)

  x = Array(typeof((1, 0.1)), sz)
  if me == root
    for k = 1:sz
      x[k] = (k-1, rand())
      @printf(fid, "Root:     (%03d, %.8e) (initial)\n", x[k][1], x[k][2])
    end
    @printf(fid, "\n")
  end

  mstscatter!(x, root, comm)

  if me == root
    @printf(fid, "Root:     (%03d, %.8e) (final)\n", x[me+1][1], x[me+1][2])
  else
    @printf(fid, "Non-Root: (%03d, %.8e (final))\n", x[me+1][1], x[me+1][2])
  end
  println(fid,"--------------------------------------")
  close(fid)
end

# gather example
function gather(root, comm)

  me = MPI.Comm_rank(comm)
  sz = MPI.Comm_size(comm)

  fid = open(@sprintf("output/collective_gather_%03d.txt", me), "w")

  println(fid,"--------------------------------------")
  @printf(fid, "Gather: root = %03d\n", root)

  val = (me, rand())
  x = Array(typeof(val), sz)
  x[me+1] = val
  if me == root
    @printf(fid, "Root:     (%03d, %.8e) (initial)\n\n", x[me+1][1], x[me+1][2])
  else
    @printf(fid, "Non-Root: (%03d, %.8e (initial))\n", x[me+1][1], x[me+1][2])
  end

  mstgather!(x, root, comm)

  if me == root
    for k = 1:sz
      @printf(fid, "Root:     (%03d, %.8e) (final)\n", x[k][1], x[k][2])
    end
  end
  println(fid,"--------------------------------------")
  close(fid)
end


function main()
  MPI.Init()

  comm = MPI.COMM_WORLD

  bcast(  mod(1, MPI.Comm_size(comm)), comm)
  reduce( mod(2, MPI.Comm_size(comm)), comm)
  gather( mod(3, MPI.Comm_size(comm)), comm)
  scatter(mod(5, MPI.Comm_size(comm)), comm)

  MPI.Finalize()
end


main()
