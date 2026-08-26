"""
Master Wok Simulation: Slower Motion + Zero Bounciness + Error Logging
"""

import asyncio
import numpy as np
import torch
import omni.usd
import omni.kit.commands
import traceback
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ROBOT_PRIM_PATH = "/World/crx10ial"  # Your FANUC robot prim path
EE_LINK_NAME = "link_6"                # End-effector link name
NUM_POPCORN = 30                       # Number of popcorn kernels
WOK_RADIUS = 0.18                      # Wok radius in meters
WOK_DEPTH = 0.13                       # Wok depth
WOK_OFFSET_X = 0.18                    # Move wok center away from arm
# ==============================================================================


# 1. Procedural 3D Open Concave Wok Dish Mesh
def create_concave_wok_mesh(stage, wok_path: str, radius: float = WOK_RADIUS, depth: float = WOK_DEPTH):
    num_rings = 10
    num_segments = 24
    vertices = []
    face_vertex_counts = []
    face_vertex_indices = []

    for i in range(num_rings + 1):
        v_angle = (i / num_rings) * (np.pi / 2.0)
        r = radius * np.sin(v_angle)
        z = depth * (1.0 - np.cos(v_angle))

        for j in range(num_segments):
            h_angle = (j / num_segments) * 2.0 * np.pi
            vertices.append(Gf.Vec3f(float(r * np.cos(h_angle)), float(r * np.sin(h_angle)), float(z)))

    for i in range(num_rings):
        for j in range(num_segments):
            next_j = (j + 1) % num_segments
            idx0 = i * num_segments + j
            idx1 = i * num_segments + next_j
            idx2 = (i + 1) * num_segments + next_j
            idx3 = (i + 1) * num_segments + j

            face_vertex_counts.append(4)
            face_vertex_indices.extend([idx0, idx1, idx2, idx3])

    mesh = UsdGeom.Mesh.Define(stage, wok_path)
    mesh.GetPointsAttr().Set(vertices)
    mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
    mesh.GetDoubleSidedAttr().Set(True)

    # Move wok away from end-effector by its radius
    wok_xform = UsdGeom.XformCommonAPI(mesh.GetPrim())
    wok_xform.SetTranslate(Gf.Vec3d(WOK_OFFSET_X, 0.0, 0.0))

    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    # Articulation links MUST use convex approximations. 
    # convexDecomposition is required since triangle meshes (none) are ignored on dynamic articulation links!
    mesh_collision.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)

    return mesh


# 2. Attach Wok & Spawn FREE Popcorn Particles in World Space
def setup_wok_and_popcorn(robot_prim_path=ROBOT_PRIM_PATH, ee_link_name=EE_LINK_NAME, num_popcorn=NUM_POPCORN):
    stage = omni.usd.get_context().get_stage()
    ee_path = f"{robot_prim_path}/{ee_link_name}"
    ee_prim = stage.GetPrimAtPath(ee_path)

    if not ee_prim.IsValid():
        print(f"[Error] End-effector link '{ee_path}' not found!")
        return False

    # A. Robustly clean up EVERYTHING from all previous script versions
    paths_to_delete = [
        "/World/PopcornParticles",            
        f"{ee_path}/Wok_Bowl/PopcornGroup",   
        f"{ee_path}/Wok_Bowl"                 
    ]
    for path in paths_to_delete:
        if stage.GetPrimAtPath(path).IsValid():
            omni.kit.commands.execute('DeletePrims', paths=[path])

    # B. Create Wok Bowl attached to link_6
    wok_path = f"{ee_path}/Wok_Bowl"
    create_concave_wok_mesh(stage, wok_path, radius=WOK_RADIUS, depth=WOK_DEPTH)

    # Apply Physics Material (Zero Bounciness, High Friction)
    material_path = "/World/PopcornMaterial"
    if not stage.GetPrimAtPath(material_path).IsValid():
        UsdShade.Material.Define(stage, material_path)
        physics_material = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(material_path))
        physics_material.CreateRestitutionAttr().Set(0.0)   # 0% bounciness
        physics_material.CreateStaticFrictionAttr().Set(0.8) # High static friction
        physics_material.CreateDynamicFrictionAttr().Set(0.6) # High dynamic friction
    popcorn_mat = UsdShade.Material(stage.GetPrimAtPath(material_path))

    # Bind material to wok to prevent bouncy walls
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(wok_path)).Bind(popcorn_mat)

    # C. Get World Transformation of link_6 at t=0
    ee_xform_matrix = omni.usd.get_world_transform_matrix(ee_prim)

    # D. Spawn Popcorn in WORLD SPACE (independent rigid bodies)
    popcorn_root = "/World/PopcornParticles"
    stage.DefinePrim(popcorn_root, "Xform")
    
    # Use a structured 3D grid to prevent overlapping (Entity Cramming)
    kernel_radius = 0.008
    spacing = 0.04 # 4cm spacing for 1.6cm diameter kernels ensures plenty of clearance
    grid_xy = 5    # 5x5 grid = 25 kernels per layer

    for i in range(num_popcorn):
        p_path = f"{popcorn_root}/kernel_{i}"
        sphere = UsdGeom.Sphere.Define(stage, p_path)
        sphere.CreateRadiusAttr().Set(kernel_radius)
        
        # Color the popcorn yellow (RGB: 1.0, 0.9, 0.1)
        sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.9, 0.1)])

        # Calculate grid position to prevent collision overlap
        layer = i // (grid_xy * grid_xy)
        remainder = i % (grid_xy * grid_xy)
        ix = remainder % grid_xy
        iy = remainder // grid_xy
        
        offset_x = (ix - grid_xy/2.0) * spacing
        offset_y = (iy - grid_xy/2.0) * spacing
        offset_z = 0.05 + layer * spacing  # Stack layers upwards

        local_x = WOK_OFFSET_X + offset_x
        local_y = offset_y
        local_z = offset_z

        # Transform local offset into WORLD coordinates
        local_pos = Gf.Vec3d(local_x, local_y, local_z)
        world_pos = ee_xform_matrix.Transform(local_pos)

        sphere_xform = UsdGeom.XformCommonAPI(sphere.GetPrim())
        sphere_xform.SetTranslate(world_pos)

        # Rigid Body Physics & Mass (~0.5g)
        UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        UsdPhysics.MassAPI.Apply(sphere.GetPrim()).CreateMassAttr().Set(0.0005)
        
        # Enable Continuous Collision Detection (CCD) to prevent clipping/tunneling
        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(sphere.GetPrim())
        physx_rb.CreateEnableCCDAttr().Set(True)

        # Bind the Physics Material to reduce bounciness
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(popcorn_mat)

    print(f"Spawned {num_popcorn} yellow popcorn kernels in a safe grid!")
    return True


# 3. Minimum-Jerk S-Curve Position Generator
def compute_scurve_wok_position(
    sim_time: float, 
    freq: float = 1.45,                 
    amp_x: float = 0.046,               
    amp_z: float = 0.065,               
    amp_pitch: float = np.radians(16),  
    phase_lag_ratio: float = 0.15   # equivalent to ~54 degrees lag
) -> tuple:
    # A. S-Curve Startup Envelope (Eliminates the initial jerk completely)
    ramp_duration = 2.0
    if sim_time < ramp_duration:
        u = sim_time / ramp_duration
        envelope = 6 * u**5 - 15 * u**4 + 10 * u**3  # Smootherstep
    else:
        envelope = 1.0

    # B. Periodic Minimum-Jerk S-Curve (Trapezoidal Velocity with filleted corners)
    def scurve_periodic(t):
        phase = (t * freq) % 1.0
        if phase < 0.5:
            x = phase * 2.0
            return 6 * x**5 - 15 * x**4 + 10 * x**3
        else:
            x = (phase - 0.5) * 2.0
            return 1.0 - (6 * x**5 - 15 * x**4 + 10 * x**3)

    # C. Calculate absolute positions (oscillating from -amp to +amp)
    pos_x = envelope * amp_x * 2.0 * (scurve_periodic(sim_time) - 0.5)
    pos_z = envelope * amp_z * 2.0 * (scurve_periodic(sim_time) - 0.5)
    pos_pitch = envelope * amp_pitch * 2.0 * (scurve_periodic(sim_time - phase_lag_ratio / freq) - 0.5)

    return pos_x, pos_z, pos_pitch


def apply_aesthetics(stage, robot_prim_path):
    from pxr import Usd, UsdShade, Gf, Sdf
    
    # Helper to create a UsdPreviewSurface material (supported by RTX renderer)
    def create_color_material(path, color_rgb, roughness=0.4):
        if stage.GetPrimAtPath(path).IsValid():
            return UsdShade.Material(stage.GetPrimAtPath(path))
        
        # Ensure /World/Looks exists
        looks_scope = "/World/Looks"
        if not stage.GetPrimAtPath(looks_scope).IsValid():
            UsdGeom.Scope.Define(stage, looks_scope)
            
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color_rgb)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.1)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    orange_mat = create_color_material("/World/Looks/Orange", Gf.Vec3f(1.0, 0.35, 0.0), 0.3)
    grey_mat = create_color_material("/World/Looks/LightGrey", Gf.Vec3f(0.6, 0.6, 0.6), 0.8)
    black_mat = create_color_material("/World/Looks/WokBlack", Gf.Vec3f(0.05, 0.05, 0.05), 0.2)

    # Color Manipulator Orange
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if robot_prim.IsValid():
        # Bind with 'strongerThanDescendants' to forcibly overwrite all child link materials!
        mat_api = UsdShade.MaterialBindingAPI.Apply(robot_prim)
        mat_api.Bind(orange_mat, UsdShade.Tokens.strongerThanDescendants)
        
    # Color Wok Black
    wok_prim = stage.GetPrimAtPath("/World/Wok_Bowl")
    if wok_prim.IsValid():
        mat_api = UsdShade.MaterialBindingAPI.Apply(wok_prim)
        mat_api.Bind(black_mat, UsdShade.Tokens.strongerThanDescendants)
    
    # Color Ground Plane Light Grey
    for prim in stage.Traverse():
        if "ground" in prim.GetPath().pathString.lower() and prim.IsA(UsdGeom.Gprim):
            mat_api = UsdShade.MaterialBindingAPI.Apply(prim)
            mat_api.Bind(grey_mat, UsdShade.Tokens.strongerThanDescendants)

# 4. Master Live Simulation Execution Loop
async def run_wok_simulation_master(prim_path: str = ROBOT_PRIM_PATH, total_seconds: float = 30.0):
    try:
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.utils.types import ArticulationAction
        import omni.timeline
        import omni.kit.app

        # Step 1. Setup Scene Geometry & Popcorn
        if not setup_wok_and_popcorn(robot_prim_path=prim_path):
            return

        # Apply Aesthetics
        stage = omni.usd.get_context().get_stage()
        apply_aesthetics(stage, prim_path)

        # Step 2. Start Physics Timeline
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            timeline.play()
            for _ in range(15):
                await omni.kit.app.get_app().next_update_async()

        # Step 3. Initialize Robot
        robot = Articulation(prim_path=prim_path, name="fanuc_robot")
        robot.initialize()
        controller = robot.get_articulation_controller()
        
        # Keep high stiffness so the robot actually follows the command instead of lagging
        controller.set_gains(
            kps=np.array([10000000.0] * robot.num_dof),
            kds=np.array([100000.0] * robot.num_dof)
        )

        base_positions = robot.get_joint_positions()
        dt = 0.02
        step = 0
        max_steps = int(total_seconds / dt)

        print("=== [Live Motion] Wok Flipping Simulation Started ===")

        # Step 4. Active Motion Loop
        while step < max_steps:
            sim_time = step * dt
            
            # Get absolute positional targets from the S-Curve generator
            pos_x, pos_z, pos_pitch = compute_scurve_wok_position(sim_time=sim_time)
            
            target_joint_positions = base_positions.copy()
            
            # Apply exact absolute positions to the joints (no arbitrary multipliers needed!)
            target_joint_positions[1] += pos_x       # Joint 2 (Forward/Back)
            target_joint_positions[2] += pos_z       # Joint 3 (Up/Down Lift)
            target_joint_positions[4] += pos_pitch   # Joint 5 (Wok Pitch Tilt)

            controller.apply_action(ArticulationAction(joint_positions=target_joint_positions))

            step += 1
            await omni.kit.app.get_app().next_update_async()

        print("=== [Live Motion] Wok Flipping Simulation Completed ===")

    except Exception as e:
        print("\n" + "="*50)
        print("CRITICAL ERROR IN WOK SIMULATION:")
        print("="*50)
        traceback.print_exc()
        print("="*50 + "\n")

# Run Master Simulation