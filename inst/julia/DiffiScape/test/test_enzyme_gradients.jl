using Test

@testset "enzyme_gradients.jl" begin

    # ====================== enzyme_available ================================

    @testset "enzyme_available returns Bool" begin
        result = enzyme_available()
        @test result isa Bool
    end

    # ====================== single_window_scalar ============================

    @testset "single_window_scalar returns scalar" begin
        cfg = SolverConfig(radius=3, block_size=1, cg_tol=1e-10)
        R_local = fill(5.0, 7, 7)
        d_cum = ones(7, 7)
        s = single_window_scalar(R_local, d_cum, 4, 4, 1, 1, cfg)
        @test s isa Float64
        @test s >= 0.0
    end

    @testset "single_window_scalar zero cotangent → zero" begin
        cfg = SolverConfig(radius=3, block_size=1, cg_tol=1e-10)
        R_local = fill(5.0, 7, 7)
        d_cum = zeros(7, 7)
        s = single_window_scalar(R_local, d_cum, 4, 4, 1, 1, cfg)
        @test s ≈ 0.0
    end

    @testset "single_window_scalar dot product consistency" begin
        # For identity cotangent, scalar = sum of current in window
        cfg = SolverConfig(radius=3, block_size=1, cg_tol=1e-10)
        R_local = fill(3.0, 7, 7)
        current = solve_single_window(R_local, 4, 4, 1, 1, cfg)

        d_cum = ones(7, 7)
        s = single_window_scalar(R_local, d_cum, 4, 4, 1, 1, cfg)
        @test s ≈ sum(current) atol=1e-8
    end

    @testset "single_window_scalar selective cotangent" begin
        # Setting d_cum to a unit vector at (4,4) should return current[4,4]
        cfg = SolverConfig(radius=3, block_size=1, cg_tol=1e-10)
        R_local = fill(3.0, 7, 7)
        current = solve_single_window(R_local, 4, 4, 1, 1, cfg)

        d_cum = zeros(7, 7)
        d_cum[4, 4] = 1.0
        s = single_window_scalar(R_local, d_cum, 4, 4, 1, 1, cfg)
        @test s ≈ current[4, 4] atol=1e-8
    end

    # ====================== resistance_gradient =============================

    @testset "resistance_gradient basic chain rule" begin
        # Simple scenario: 4 cells, 2 basis functions
        # R = clamp(exp(r0 + z1*phi1 + z2*phi2), R_min, R_max)
        cfg = SolverConfig(R_min=0.1, R_max=1000.0)
        params = [1.0, 0.5, -0.3]  # [r0, z1, z2]
        basis_values = [0.1 0.2; 0.3 0.4; 0.5 0.6; 0.7 0.8]

        # Compute R_flat
        n_cells = 4
        R_flat = zeros(n_cells)
        for i in 1:n_cells
            log_r = params[1] + params[2] * basis_values[i, 1] + params[3] * basis_values[i, 2]
            R_flat[i] = clamp(exp(log_r), cfg.R_min, cfg.R_max)
        end

        # Synthetic dR
        dR_flat = [1.0, 2.0, 3.0, 4.0]

        grad = resistance_gradient(params, basis_values, dR_flat, R_flat, cfg)
        @test length(grad) == 3

        # Manual check: grad[1] = sum(dR_flat .* R_flat) for unclamped cells
        expected_r0 = sum(dR_flat .* R_flat)
        @test grad[1] ≈ expected_r0 atol=1e-10

        # grad[2] = sum(dR_flat .* R_flat .* basis_values[:, 1])
        expected_z1 = sum(dR_flat .* R_flat .* basis_values[:, 1])
        @test grad[2] ≈ expected_z1 atol=1e-10

        expected_z2 = sum(dR_flat .* R_flat .* basis_values[:, 2])
        @test grad[3] ≈ expected_z2 atol=1e-10
    end

    @testset "resistance_gradient clamped cells contribute zero" begin
        cfg = SolverConfig(R_min=1.0, R_max=10.0)
        params = [0.0]  # only r0, no basis
        basis_values = reshape(Float64[], 3, 0)

        # R = exp(0) = 1.0 for all cells → exactly at R_min → clamped
        R_flat = [1.0, 5.0, 10.0]  # cells at min, mid, max
        dR_flat = [1.0, 1.0, 1.0]

        grad = resistance_gradient(params, basis_values, dR_flat, R_flat, cfg)
        # Only the middle cell (R=5, strictly between min and max) contributes
        @test grad[1] ≈ 5.0 atol=1e-10
    end

    @testset "resistance_gradient zero dR → zero gradient" begin
        cfg = SolverConfig(R_min=0.1, R_max=1000.0)
        params = [1.0, 0.5]
        basis_values = reshape([0.1, 0.3, 0.5, 0.7], 4, 1)
        R_flat = exp.(1.0 .+ 0.5 .* [0.1, 0.3, 0.5, 0.7])
        dR_flat = zeros(4)

        grad = resistance_gradient(params, basis_values, dR_flat, R_flat, cfg)
        @test all(grad .== 0.0)
    end

    # ====================== resistance_gradient_generic =======================

    @testset "resistance_gradient_generic matches resistance_gradient (exp link)" begin
        cfg = SolverConfig(R_min=0.1, R_max=1000.0)
        params = [1.0, 0.5, -0.3]
        basis_values = [0.1 0.2; 0.3 0.4; 0.5 0.6; 0.7 0.8]

        n_cells = 4
        R_flat = zeros(n_cells)
        for i in 1:n_cells
            log_r = params[1] + params[2] * basis_values[i, 1] + params[3] * basis_values[i, 2]
            R_flat[i] = clamp(exp(log_r), cfg.R_min, cfg.R_max)
        end

        dR_flat = [1.0, 2.0, 3.0, 4.0]

        grad_old = resistance_gradient(params, basis_values, dR_flat, R_flat, cfg)

        # Build dR_deta the same way the old code did internally
        dR_deta = zeros(n_cells)
        for i in 1:n_cells
            ri = R_flat[i]
            if ri > cfg.R_min && ri < cfg.R_max
                dR_deta[i] = ri
            end
        end

        grad_new = resistance_gradient_generic(basis_values, dR_flat, dR_deta)
        @test grad_new ≈ grad_old atol=1e-10
    end

    @testset "resistance_gradient_generic with identity link (dR/deta = 1)" begin
        basis_values = reshape([0.2, 0.5, 0.8], 3, 1)
        dR_flat = [1.0, 2.0, 3.0]
        dR_deta = [1.0, 1.0, 1.0]  # identity link: dR/deta = 1

        grad = resistance_gradient_generic(basis_values, dR_flat, dR_deta)
        @test length(grad) == 2
        @test grad[1] ≈ sum(dR_flat) atol=1e-10
        @test grad[2] ≈ sum(dR_flat .* basis_values[:, 1]) atol=1e-10
    end

    @testset "resistance_gradient_generic with zero dR_deta at clamped cells" begin
        basis_values = reshape([0.1, 0.3, 0.5], 3, 1)
        dR_flat = [1.0, 2.0, 3.0]
        dR_deta = [0.0, 5.0, 0.0]  # only middle cell unclamped

        grad = resistance_gradient_generic(basis_values, dR_flat, dR_deta)
        @test grad[1] ≈ 2.0 * 5.0 atol=1e-10
        @test grad[2] ≈ 2.0 * 5.0 * 0.3 atol=1e-10
    end

    # ====================== enzyme_gradient_generic (skip if Enzyme unavailable)

    @testset "enzyme_gradient_generic smoke test" begin
        if !enzyme_available()
            @info "Skipping enzyme_gradient_generic tests — Enzyme.jl not available"
        else
            cfg = SolverConfig(radius=2, block_size=1, cg_tol=1e-8)
            R_flat = fill(5.0, 25)
            dR_deta = fill(5.0, 25)  # exp link: dR/deta = R
            basis_values = reshape(Float64[], 25, 0)

            grad = enzyme_gradient_generic(R_flat, dR_deta, basis_values,
                                            5, 5, cfg)
            @test length(grad) == 1
            @test isfinite(grad[1])
        end
    end

    # ====================== enzyme_gradient (skip if Enzyme unavailable) ====

    @testset "enzyme_gradient smoke test" begin
        if !enzyme_available()
            @info "Skipping enzyme_gradient tests — Enzyme.jl not available"
        else
            cfg = SolverConfig(radius=2, block_size=1, cg_tol=1e-8)
            params = [log(5.0)]  # r0 only, no basis
            basis_values = reshape(Float64[], 25, 0)  # 5×5 grid

            grad = enzyme_gradient(params, basis_values, 5, 5, cfg)
            @test length(grad) == 1
            @test isfinite(grad[1])
        end
    end

    # ====================== enzyme_hvp (skip if Enzyme unavailable) =========

    @testset "enzyme_hvp smoke test" begin
        if !enzyme_available()
            @info "Skipping enzyme_hvp tests — Enzyme.jl not available"
        else
            cfg = SolverConfig(radius=2, block_size=1, cg_tol=1e-8)
            params = [log(5.0)]
            basis_values = reshape(Float64[], 25, 0)
            v = [1.0]

            hvp = enzyme_hvp(params, v, basis_values, 5, 5, cfg)
            @test length(hvp) == 1
            @test isfinite(hvp[1])
        end
    end

end
