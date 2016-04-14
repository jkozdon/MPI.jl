#=
Parallel sort based on the Baudet–Stevenson odd–even merge-splitting algorithm
@techreport{baudet1975optimal,
  title = {Optimal sorting algorithms for parallel computers},
  author = {G. Baudet and D. Stevenson},
  year = {1975},
  month = {may},
  institution = {Computer Science Department, Carnegie-Mellon University}
}

See also: https://en.wikipedia.org/wiki/Odd%E2%80%93even_sort
=#

function merge!(rk_x, nb_x, tmp, rk, nb, comm)

  # Number of elements
  N = length(rk_x)

  # Exchange the data
  rreq = MPI.Irecv!(nb_x, nb, 33, comm)
  sreq = MPI.Isend(rk_x, nb, 33, comm)
  MPI.Waitall!([rreq, sreq])

  # Do the merge
  rk_k = 1
  nb_k = 1
  for k = 1:length(tmp)

    # If nb is done OR rk is not done and rk value is smaller
    if (nb_k > N) || (rk_k <= N && rk_x[rk_k] < nb_x[nb_k])
      tmp[k] = rk_x[rk_k]
      rk_k += 1
    else
      tmp[k] = nb_x[nb_k]
      nb_k += 1
    end

  end

  # Keep the start of finish of tmp?
  if rk < nb
    rk_x[:] = tmp[1:N]
  else
    rk_x[:] = tmp[N+1:end]
  end
end

function psort!(my_x::Array, comm)

  # This is temp storage for my neighbor values and the merge
  nb_x = similar(my_x)
  tmp = Array(typeof(my_x[1]), 2 * length(my_x))

  # Get the MPI rank and size
  rk = MPI.Comm_rank(comm)
  sz = MPI.Comm_size(comm)

  # We need to merge MPI comm size times
  for k = 1:sz

    # Determine neighbor
    if mod(k, 2) == mod(rk, 2)
      nb = rk - 1
    else
      nb = rk + 1
    end

    # merge with neighbor
    if nb >= 0 && nb < sz
      merge!(my_x, nb_x, tmp, rk, nb, comm)
    end

  end

end
