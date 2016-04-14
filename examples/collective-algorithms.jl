#=
These are implementations of collective communication algorithms from
@article{chan2007collective,
  title={Collective communication: theory, practice, and experience},
  author={Chan, Ernie and Heimlich, Marcel and Purkayastha, Avi and
          Van De Geijn, Robert},
  journal={Concurrency and Computation: Practice and Experience},
  volume={19},
  number={13},
  pages={1749--1783},
  year={2007},
  publisher={Wiley Online Library}
}
=#

# MST Broadcat Figure 3(a)
function mstbcast!(x, root, comm)
  me = MPI.Comm_rank(comm)
  sz = MPI.Comm_size(comm)
  mstbcast!(x, root, 0, sz-1, me, comm)
end
function mstbcast!(x, root, left, right, me, comm)

  # If I'm the only one in the tree then return
  if left == right
    return
  end

  # Integer division to find the splitting point
  mid = div(left + right, 2)

  # If the root is in the left half then dest is the far right
  # otherwise the dest if the far left
  if root <= mid
    dest = right
  else
    dest = left
  end

  # If I'm the root I have the data to send
  if me == root
    sreq = MPI.Isend( x, dest, 999, comm)
  end

  # If I'm the dest I need to receive the data (I cannot continue until I have
  # my data so we block here)
  if me == dest
    # MPI.Recv!(x, root, 999, comm)
    rreq = MPI.Irecv!(x, root, 999, comm)
    MPI.Wait!(rreq)
  end

  # Determine how the recursively call the broadcast function
  if me <= mid && root <= mid
    # If I'm on the left and the root is on the left the root is my root
    mstbcast!(x, root, left, mid, me, comm)

  elseif me <= mid && root > mid
    # If I'm on the left but the root is on the right, then dest is my root
    mstbcast!(x, dest, left, mid, me, comm)

  elseif me > mid && root <= mid
    # If I'm on the right but the root is on the left, then dest is my root
    mstbcast!(x, dest, mid+1, right, me, comm)

  elseif me > mid && root > mid
    # If I'm on the right and the root is on the right the root is my root
    mstbcast!(x, root, mid+1, right, me, comm)

  end

  if me == root
    MPI.Wait!(sreq)
  end
end

# MST Reduce Figure 3(b)
function mstreduce!(x, op, root, comm)
  me = MPI.Comm_rank(comm)
  sz = MPI.Comm_size(comm)
  mstreduce!(x, op, root, 0, sz-1, me, comm)
end
function mstreduce!(x, op, root, left, right, me, comm)

  # If I'm the only one in the tree then return
  if left == right
    return
  end

  # Integer division to find the splitting point
  mid = div(left + right, 2)

  # If the root is in the left half then srce is the far right
  # otherwise the srce if the far left
  if root <= mid
    srce = right
  else
    srce = left
  end

  # post receive as soon as we can
  if me == root
    tmp = similar(x) # Like a copy, but without the copy of the data
    rreq = MPI.Irecv!(tmp, srce, 999, comm)
  end

  # Determine how the recursively call the broadcast function
  if me <= mid && root <= mid
    # If I'm on the left and the root is on the left the root is my root
    mstreduce!(x, op, root, left, mid, me, comm)

  elseif me <= mid && root > mid
    # If I'm on the left but the root is on the right, then srce is my root
    mstreduce!(x, op, srce, left, mid, me, comm)

  elseif me > mid && root <= mid
    # If I'm on the right but the root is on the left, then srce is my root
    mstreduce!(x, op, srce, mid+1, right, me, comm)

  elseif me > mid && root > mid
    # If I'm on the right and the root is on the right the root is my root
    mstreduce!(x, op, root, mid+1, right, me, comm)
  end

  # If I'm the srce and I have the data to send (doing this non-blocking cannot
  # help as all)
  if me == srce
    sreq = MPI.Send( x, root, 999, comm)
  end

  if me == root
    MPI.Wait!(rreq)
    x[:] = op(x, tmp)
  end
end

# MST Scatter Figure 3(e)
function mstscatter!(x, root, comm)
  me = MPI.Comm_rank(comm)
  sz = MPI.Comm_size(comm)
  mstscatter!(x, root, 0, sz-1, me, comm)
end
function mstscatter!(x, root, left, right, me, comm)

  # If I'm the only one in the tree then return
  if left == right
    return
  end

  # Integer division to find the splitting point
  mid = div(left + right, 2)

  # If the root is in the left half then dest is the far right
  # otherwise the dest if the far left
  if root <= mid
    dest = right
  else
    dest = left
  end

  # Post the recv before recursing
  if root <= mid
    # If root is on the left we are sending the bottom
    if me == root
      sreq = MPI.Isend(Ref(x,mid+2), right - mid, dest, 999, comm)
    end
    if me == dest
      MPI.Recv!(Ref(x,mid+2), right - mid, root, 999, comm)
    end
  else
    # If root is on the right we are sending the top
    if me == root
      sreq = MPI.Isend(Ref(x,left+1), mid - left + 1, dest, 999, comm)
    end
    if me == dest
      MPI.Recv!(Ref(x,left+1), mid - left + 1, root, 999, comm)
    end
  end

  # Determine how the recursively call the broadcast function
  if me <= mid && root <= mid
    # If I'm on the left and the root is on the left the root is my root
    mstscatter!(x, root, left, mid, me, comm)

  elseif me <= mid && root > mid
    # If I'm on the left but the root is on the right, then dest is my root
    mstscatter!(x, dest, left, mid, me, comm)

  elseif me > mid && root <= mid
    # If I'm on the right but the root is on the left, then dest is my root
    mstscatter!(x, dest, mid+1, right, me, comm)

  elseif me > mid && root > mid
    # If I'm on the right and the root is on the right the root is my root
    mstscatter!(x, root, mid+1, right, me, comm)
  end

  # Wait on the send ofter recursion
  if me == root
    MPI.Wait!(sreq)
  end
end

# MST Gather Figure 3(d)
function mstgather!(x, root, comm)
  me = MPI.Comm_rank(comm)
  sz = MPI.Comm_size(comm)
  mstgather!(x, root, 0, sz-1, me, comm)
end
function mstgather!(x, root, left, right, me, comm)

  # If I'm the only one in the tree then return
  if left == right
    return
  end

  # Integer division to find the splitting point
  mid = div(left + right, 2)

  # If the root is in the left half then srce is the far right
  # otherwise the srce if the far left
  if root <= mid
    srce = right
  else
    srce = left
  end

  # Post the recv before recursing
  if me == root && root <= mid
    # If root is on the left we are filling the bottom
    rreq = MPI.Irecv!(Ref(x,mid+2), right - mid, srce, 999, comm)
  elseif me == root
    # If root is on the right we are filling the top
    rreq = MPI.Irecv!(Ref(x,left+1), mid - left + 1, srce, 999, comm)
  end

  # Determine how the recursively call the broadcast function
  if me <= mid && root <= mid
    # If I'm on the left and the root is on the left the root is my root
    mstgather!(x, root, left, mid, me, comm)

  elseif me <= mid && root > mid
    # If I'm on the left but the root is on the right, then srce is my root
    mstgather!(x, srce, left, mid, me, comm)

  elseif me > mid && root <= mid
    # If I'm on the right but the root is on the left, then srce is my root
    mstgather!(x, srce, mid+1, right, me, comm)

  elseif me > mid && root > mid
    # If I'm on the right and the root is on the right the root is my root
    mstgather!(x, root, mid+1, right, me, comm)
  end

  if root <= mid &&  me == srce
    # If root is on the left we are filling the bottom
    MPI.Send(Ref(x,mid+2), right - mid, root, 999, comm)
  elseif me == srce
    # If root is on the right we are filling the top
    MPI.Send(Ref(x,left+1), mid - left + 1, root, 999, comm)
  end

  # Wait on the recv ofter recursion
  if me == root
    MPI.Wait!(rreq)
  end
end
