using Test

@testset "differentiable_solver.jl" begin

    # ====================== SolverConfig ====================================

    @testset "SolverConfig defaults" begin
        cfg = SolverConfig()
        @test cfg.radius == 13
        @test cfg.block_size == 5
        @test cfg.source_from_resistance == true
        @test cfg.R_min == 1.0
        @test cfg.R_max == 5000.0
        @test cfg.cg_tol == 1e-8
        @test cfg.cg_maxiter == 2000
        @test cfg.ground_conductance == 1e10
    end

    @testset "SolverConfig custom" begin
        cfg = SolverConfig(radius=5, block_size=3, R_min=0.5, R_max=100.0,
                           cg_tol=1e-6, cg_maxiter=500)
        @test cfg.radius == 5
        @test cfg.block_size == 3
        @test cfg.R_min == 0.5
        @test cfg.R_max == 100.0
    end

    # ====================== resistance_to_conductance =======================

    @testset "resistance_to_conductance uniform" begin
        R = fill(4.0, 3, 4)
        eh, ev = resistance_to_conductance(R)
        @test size(eh) == (3, 3)
        @test size(ev) == (2, 4)
        # Harmonic mean of equal values: 2/(4+4)=0.25
        @test all(eh .≈ 0.25)
        @test all(ev .≈ 0.25)
    end

    @testset "resistance_to_conductance simple pair" begin
        R = [2.0 8.0]
        eh, ev = resistance_to_conductance(R)
        @test size(eh) == (1, 1)
        @test eh[1, 1] ≈ 2.0 / (2.0 + 8.0)  # = 0.2
        @test size(ev) == (0, 2)
    end

    @testset "resistance_to_conductance zeros & negatives" begin
        R = [1.0 0.0; -1.0 2.0]
        eh, ev = resistance_to_conductance(R)
        # (1,1)-(1,2): R[1,2]=0 → 0
        @test eh[1, 1] == 0.0
        # (2,1)-(2,2): R[2,1]=-1 → 0
        @test eh[2, 1] == 0.0
        # (1,1)-(2,1): R[2,1]=-1 → 0
        @test ev[1, 1] == 0.0
        # (1,2)-(2,2): R[1,2]=0 → 0
        @test ev[1, 2] == 0.0
    end

    @testset "resistance_to_conductance NaN/Inf" begin
        R = [1.0 NaN; Inf 2.0]
        eh, ev = resistance_to_conductance(R)
        @test eh[1, 1] == 0.0   # NaN neighbor
        @test eh[2, 1] == 0.0   # Inf neighbor
        @test ev[1, 1] == 0.0   # Inf neighbor
        @test ev[1, 2] == 0.0   # NaN neighbor
    end

    @testset "resistance_to_conductance single cell" begin
        R = fill(5.0, 1, 1)
        eh, ev = resistance_to_conductance(R)
        @test size(eh) == (1, 0)
        @test size(ev) == (0, 1)
    end

    # ====================== laplacian_matvec! ===============================

    @testset "laplacian_matvec! constant voltage → zero" begin
        # Constant x → Lx = 0 (no ground), since all neighbors equal
        R = fill(2.0, 4, 4)
        eh, ev = resistance_to_conductance(R)
        gm = falses(4, 4)
        x = fill(3.0, 4, 4)
        y = zeros(4, 4)
        laplacian_matvec!(y, x, eh, ev, gm, 1e10)
        @test all(y .≈ 0.0)
    end

    @testset "laplacian_matvec! ground adds diagonal" begin
        R = fill(2.0, 2, 2)
        eh, ev = resistance_to_conductance(R)
        gm = trues(2, 2)
        g_cond = 100.0
        x = fill(1.0, 2, 2)
        y = zeros(2, 2)
        laplacian_matvec!(y, x, eh, ev, gm, g_cond)
        # The Laplacian part vanishes for constant x; ground gives g_cond*x
        @test all(y .≈ g_cond)
    end

    @testset "laplacian_matvec! 1D chain" begin
        # A 1×3 grid: cells 1-2-3, R=1 everywhere
        # eh: G12=1, G23=1
        # x = [0, 1, 0]
        # L*x at cell 2: G12*(1-0) + G23*(1-0) = 2
        # L*x at cell 1: G12*(0-1) = -1
        # L*x at cell 3: G23*(0-1) = -1
        R = fill(1.0, 1, 3)
        eh, ev = resistance_to_conductance(R)
        gm = falses(1, 3)
        x = [0.0 1.0 0.0]
        y = zeros(1, 3)
        laplacian_matvec!(y, x, eh, ev, gm, 0.0)
        @test y[1, 1] ≈ -1.0
        @test y[1, 2] ≈ 2.0
        @test y[1, 3] ≈ -1.0
    end

    @testset "laplacian_matvec! symmetry" begin
        # L should be symmetric: dot(x, Ly) == dot(y, Lx)
        R = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
        eh, ev = resistance_to_conductance(R)
        gm = falses(3, 3)
        gm[1, 1] = true  # ground one cell
        g_cond = 50.0

        x = randn(3, 3)
        y_vec = randn(3, 3)
        Lx = zeros(3, 3)
        Ly = zeros(3, 3)
        laplacian_matvec!(Lx, x, eh, ev, gm, g_cond)
        laplacian_matvec!(Ly, y_vec, eh, ev, gm, g_cond)

        @test sum(x .* Ly) ≈ sum(y_vec .* Lx) atol=1e-10
    end

    # ====================== compute_diag_inv! ===============================

    @testset "compute_diag_inv! matches Lx diagonal" begin
        R = [2.0 3.0; 5.0 7.0]
        eh, ev = resistance_to_conductance(R)
        gm = falses(2, 2)
        gm[1, 2] = true
        g_cond = 100.0

        diag_inv = zeros(2, 2)
        compute_diag_inv!(diag_inv, eh, ev, gm, g_cond)

        # Verify against laplacian_matvec! with unit vectors
        for r in 1:2, c in 1:2
            e = zeros(2, 2)
            e[r, c] = 1.0
            Le = zeros(2, 2)
            laplacian_matvec!(Le, e, eh, ev, gm, g_cond)
            d_ii = Le[r, c]
            if d_ii > 0
                @test diag_inv[r, c] ≈ 1.0 / d_ii atol=1e-12
            end
        end
    end

    # ====================== cg_solve ========================================

    @testset "cg_solve zero RHS" begin
        R = fill(2.0, 3, 3)
        eh, ev = resistance_to_conductance(R)
        gm = trues(3, 3)
        b = zeros(3, 3)
        v = cg_solve(b, eh, ev, gm, 1e10)
        @test all(abs.(v) .< 1e-12)
    end

    @testset "cg_solve known 1D system" begin
        # 1×3 grid, R=1 everywhere, ground at boundary cells 1 & 3
        # Inject 1 amp at cell 2
        # L: 2×2 harmonic mean = 1.0 for each edge
        # Diagonal: cell 1: G12 + g = 1+G, cell 2: G12+G23 = 2, cell 3: G23+g = 1+G
        # Solve (L + g·D_g) v = b, b = [0, 1, 0]
        R = fill(1.0, 1, 3)
        eh, ev = resistance_to_conductance(R)
        gm = Bool[true false true]
        g_cond = 1e10
        b = Float64[0.0 1.0 0.0]
        v = cg_solve(b, eh, ev, gm, g_cond; tol=1e-12, maxiter=100)

        # Ground nodes should have voltage ≈ 0
        @test abs(v[1, 1]) < 1e-6
        @test abs(v[1, 3]) < 1e-6
        # Center node should have positive voltage
        @test v[1, 2] > 0
    end

    @testset "cg_solve Lv = b residual small" begin
        # Build a random problem and check ||Lv - b|| is small
        R = 1.0 .+ 4.0 * rand(5, 5)
        eh, ev = resistance_to_conductance(R)
        gm = falses(5, 5)
        gm[1, :] .= true
        gm[5, :] .= true
        gm[:, 1] .= true
        gm[:, 5] .= true
        g_cond = 1e6
        b = randn(5, 5)
        v = cg_solve(b, eh, ev, gm, g_cond; tol=1e-10, maxiter=5000)

        Lv = zeros(5, 5)
        laplacian_matvec!(Lv, v, eh, ev, gm, g_cond)
        residual = maximum(abs.(Lv .- b))
        bnorm = sqrt(sum(b .^ 2))
        @test residual / bnorm < 1e-6
    end

    # ====================== compute_current =================================

    @testset "compute_current uniform voltage → zero" begin
        v = fill(5.0, 3, 3)
        R = fill(2.0, 3, 3)
        eh, ev = resistance_to_conductance(R)
        I = compute_current(v, eh, ev)
        @test all(I .≈ 0.0)
    end

    @testset "compute_current 1D linear voltage" begin
        # 1×3, R=1 → G=1 for each edge
        # v = [0, 1, 2]: current through each edge = |G*(v_i - v_j)| = 1
        # cell 1: 0.5*|1*(0-1)| = 0.5
        # cell 2: 0.5*(|1*(1-0)| + |1*(1-2)|) = 1.0
        # cell 3: 0.5*|1*(2-1)| = 0.5
        v = Float64[0.0 1.0 2.0]
        R = fill(1.0, 1, 3)
        eh, ev = resistance_to_conductance(R)
        I = compute_current(v, eh, ev)
        @test I[1, 1] ≈ 0.5
        @test I[1, 2] ≈ 1.0
        @test I[1, 3] ≈ 0.5
    end

    @testset "compute_current non-negative" begin
        v = randn(5, 5)
        R = 1.0 .+ 4.0 * rand(5, 5)
        eh, ev = resistance_to_conductance(R)
        I = compute_current(v, eh, ev)
        @test all(I .>= 0.0)
    end

    # ====================== compute_block_centers ===========================

    @testset "compute_block_centers basic" begin
        centers = compute_block_centers(10, 10, 5)
        # First center at (3, 3), then (3,8), (8,3), (8,8)
        @test (3, 3) in centers
        @test (3, 8) in centers
        @test (8, 3) in centers
        @test (8, 8) in centers
        @test length(centers) == 4
    end

    @testset "compute_block_centers block_size 1" begin
        centers = compute_block_centers(3, 3, 1)
        # half=0, start at 1, step 1 → (1,1),(1,2),(1,3),(2,1),...,(3,3)
        @test length(centers) == 9
    end

    @testset "compute_block_centers tiny grid" begin
        centers = compute_block_centers(2, 2, 5)
        # half=2, start at 3 → exceeds grid → empty
        @test length(centers) == 0
    end

    # ====================== window_bounds ===================================

    @testset "window_bounds no clipping" begin
        r1, r2, c1, c2 = window_bounds(10, 10, 5, 20, 20)
        @test r1 == 5
        @test r2 == 15
        @test c1 == 5
        @test c2 == 15
    end

    @testset "window_bounds clipped at edges" begin
        r1, r2, c1, c2 = window_bounds(2, 2, 5, 10, 10)
        @test r1 == 1
        @test r2 == 7
        @test c1 == 1
        @test c2 == 7
    end

    @testset "window_bounds corner" begin
        r1, r2, c1, c2 = window_bounds(1, 1, 3, 5, 5)
        @test r1 == 1
        @test r2 == 4
        @test c1 == 1
        @test c2 == 4
    end

    # ====================== prepare_window! =================================

    @testset "prepare_window! center block marked as source" begin
        cfg = SolverConfig(radius=5, block_size=3)
        wr, wc = 11, 11  # 2*radius+1
        R_win = fill(10.0, wr, wc)
        gm = falses(wr, wc)
        sm = falses(wr, wc)
        cr, cc = 6, 6  # center of window in global coords
        r1, c1 = 1, 1
        prepare_window!(R_win, gm, sm, cr, cc, r1, c1, wr, wc, 5, 3)

        # Center block: rows/cols in [5,7] (where cr=6, half=1)
        @test sm[6, 6] == true
        @test sm[5, 5] == true
        @test sm[7, 7] == true
        # Outside block
        @test sm[4, 4] == false
    end

    @testset "prepare_window! outside circle is grounded/zeroed" begin
        cfg = SolverConfig(radius=3, block_size=1)
        wr, wc = 7, 7
        R_win = fill(5.0, wr, wc)
        gm = falses(wr, wc)
        sm = falses(wr, wc)
        prepare_window!(R_win, gm, sm, 4, 4, 1, 1, wr, wc, 3, 1)

        # Corners are outside circle (dist > 3)
        @test R_win[1, 1] == 0.0
        @test gm[1, 1] == true
        @test R_win[1, 7] == 0.0
        @test gm[1, 7] == true
    end

    @testset "prepare_window! boundary ring is grounded" begin
        cfg = SolverConfig(radius=4, block_size=1)
        wr, wc = 9, 9
        R_win = fill(5.0, wr, wc)
        gm = falses(wr, wc)
        sm = falses(wr, wc)
        prepare_window!(R_win, gm, sm, 5, 5, 1, 1, wr, wc, 4, 1)

        # Cell at distance exactly radius (4,0): (5,9) → dr=0, dc=4, dist=4
        # dist_sq=16 == radius_sq=16 → in_circle, but dist_sq > inner(9) → grounded
        @test gm[5, 9] == true
        @test R_win[5, 9] == 5.0  # R kept for boundary
    end

    @testset "prepare_window! invalid cells zeroed" begin
        wr, wc = 5, 5
        R_win = fill(5.0, wr, wc)
        R_win[3, 3] = 0.0  # invalid cell at center
        gm = falses(wr, wc)
        sm = falses(wr, wc)
        prepare_window!(R_win, gm, sm, 3, 3, 1, 1, wr, wc, 2, 1)

        @test R_win[3, 3] == 0.0
        @test gm[3, 3] == true
        @test sm[3, 3] == false
    end

    # ====================== build_source_vector =============================

    @testset "build_source_vector normalisation" begin
        sm = Bool[false true false; true true false; false false false]
        R_win = fill(2.0, 3, 3)
        b = build_source_vector(sm, R_win, true)
        @test sum(b) ≈ 1.0
        # Only source cells are nonzero
        @test b[1, 1] == 0.0
        @test b[1, 2] > 0.0
    end

    @testset "build_source_vector uniform when not from resistance" begin
        sm = falses(3, 3)
        sm[1, 1] = true
        sm[2, 2] = true
        R_win = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
        b = build_source_vector(sm, R_win, false)
        # Both source cells equal (uniform), total = 1
        @test b[1, 1] == b[2, 2]
        @test sum(b) ≈ 1.0
    end

    @testset "build_source_vector no sources" begin
        sm = falses(3, 3)
        R_win = fill(2.0, 3, 3)
        b = build_source_vector(sm, R_win, true)
        @test all(b .== 0.0)
    end

    # ====================== solve_single_window =============================

    @testset "solve_single_window non-negative current" begin
        cfg = SolverConfig(radius=3, block_size=1, cg_tol=1e-10)
        R = fill(5.0, 7, 7)
        current = solve_single_window(R, 4, 4, 1, 1, cfg)
        @test size(current) == (7, 7)
        @test all(current .>= 0.0)
        @test maximum(current) > 0.0  # should have some current
    end

    @testset "solve_single_window no valid cells returns zeros" begin
        # All NaN resistance → all cells grounded, no sources
        cfg = SolverConfig(radius=2, block_size=1)
        R = fill(NaN, 5, 5)
        current = solve_single_window(R, 3, 3, 1, 1, cfg)
        @test all(current .== 0.0)
    end

    @testset "solve_single_window higher R → less current" begin
        cfg = SolverConfig(radius=3, block_size=1, cg_tol=1e-10)
        R_low = fill(1.0, 7, 7)
        R_high = fill(100.0, 7, 7)
        I_low = solve_single_window(R_low, 4, 4, 1, 1, cfg)
        I_high = solve_single_window(R_high, 4, 4, 1, 1, cfg)
        # Higher resistance should mean lower total current flow
        @test sum(I_high) < sum(I_low)
    end

    # ====================== cumulative_current ==============================

    @testset "cumulative_current small grid" begin
        cfg = SolverConfig(radius=3, block_size=3)
        R = fill(5.0, 10, 10)
        C = cumulative_current(R, cfg)
        @test size(C) == (10, 10)
        @test all(C .>= 0.0)
        @test sum(C) > 0.0
    end

    @testset "cumulative_current all nodata" begin
        cfg = SolverConfig(radius=3, block_size=3)
        R = fill(0.0, 10, 10)  # all invalid
        C = cumulative_current(R, cfg)
        @test all(C .== 0.0)
    end

    @testset "cumulative_current R bridge wrapper" begin
        R = fill(5.0, 10, 10)
        C = cumulative_current(R, 3, 3)
        @test size(C) == (10, 10)
        @test sum(C) > 0.0
    end

    @testset "cumulative_current symmetric for symmetric R" begin
        cfg = SolverConfig(radius=3, block_size=3)
        R = fill(5.0, 9, 9)
        C = cumulative_current(R, cfg)
        # Uniform resistance → symmetric current map
        # Check horizontal symmetry
        @test C ≈ C[:, end:-1:1] atol=1e-10
        # Check vertical symmetry
        @test C ≈ C[end:-1:1, :] atol=1e-10
    end

    @testset "cumulative_current empty grid" begin
        cfg = SolverConfig(radius=3, block_size=3)
        # Grid too small for any block center
        R = fill(5.0, 2, 2)
        C = cumulative_current(R, cfg)
        @test all(C .== 0.0)
    end

end
