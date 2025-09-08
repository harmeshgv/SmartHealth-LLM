import argparse
import os
import json
from gravixlayer import GravixLayer


def parse_gpu_spec(gpu_type, gpu_count=1):
    """Parse GPU specification and return hardware string"""
    gpu_mapping = {
        "t4": "nvidia-t4-16gb-pcie_1",
        "t4": "nvidia-t4-16gb-pcie_2",
    }

    gpu_key = gpu_type.lower()
    if gpu_key not in gpu_mapping:
        raise ValueError(
            f"Unsupported GPU type: {gpu_type}. Supported: {list(gpu_mapping.keys())}")

    return f"{gpu_mapping[gpu_key]}_{gpu_count}"


def main():
    parser = argparse.ArgumentParser(
        description="GravixLayer CLI – Chat Completions, Text Completions, and Deployment Management"
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Chat/Completions parser (default behavior)
    chat_parser = subparsers.add_parser("chat", help="Chat completions")
    chat_parser.add_argument("--api-key", type=str,
                             default=None, help="API key")
    chat_parser.add_argument("--model", required=True, help="Model name")
    chat_parser.add_argument("--system", default=None,
                             help="System prompt (optional)")
    chat_parser.add_argument("--user", help="User prompt/message (chat mode)")
    chat_parser.add_argument(
        "--prompt", help="Direct prompt (completions mode)")
    chat_parser.add_argument("--temperature", type=float,
                             default=None, help="Temperature")
    chat_parser.add_argument("--max-tokens", type=int,
                             default=None, help="Maximum tokens to generate")
    chat_parser.add_argument(
        "--stream", action="store_true", help="Stream output")
    chat_parser.add_argument(
        "--mode", choices=["chat", "completions"], default="chat", help="API mode")

    # Deployments parser (for deployment management)
    deployments_parser = subparsers.add_parser(
        "deployments", help="Deployment management")
    deployments_subparsers = deployments_parser.add_subparsers(
        dest="deployments_action", help="Deployment actions")

    # Create deployment
    create_parser = deployments_subparsers.add_parser(
        "create", help="Create a new deployment")
    create_parser.add_argument(
        "--api-key", type=str, default=None, help="API key")
    create_parser.add_argument(
        "--deployment_name", required=True, help="Deployment name")
    create_parser.add_argument(
        "--hw_type", default="dedicated", help="Hardware type (default: dedicated)")
    create_parser.add_argument("--hardware", required=True,
                               help="Hardware specification (e.g., nvidia-t4-16gb-pcie_1)")
    create_parser.add_argument(
        "--min_replicas", type=int, default=1, help="Minimum replicas")
    create_parser.add_argument(
        "--model_name", required=True, help="Model name to deploy")
    create_parser.add_argument("--auto-retry", action="store_true", 
                               help="Auto-retry with unique name if deployment name exists")
    create_parser.add_argument("--wait", action="store_true",
                               help="Wait for deployment to be ready before exiting")

    # List deployments
    list_parser = deployments_subparsers.add_parser(
        "list", help="List all deployments")
    list_parser.add_argument("--api-key", type=str,
                             default=None, help="API key")
    list_parser.add_argument(
        "--json", action="store_true", help="Output as JSON")

    # Delete deployment
    delete_parser = deployments_subparsers.add_parser(
        "delete", help="Delete a deployment")
    delete_parser.add_argument(
        "--api-key", type=str, default=None, help="API key")
    delete_parser.add_argument("deployment_id", help="Deployment ID to delete")

    # Hardware/GPU listing
    hardware_parser = deployments_subparsers.add_parser(
        "hardware", help="List available hardware/GPUs")
    hardware_parser.add_argument(
        "--api-key", type=str, default=None, help="API key")
    hardware_parser.add_argument(
        "--list", action="store_true", help="List available hardware")
    hardware_parser.add_argument(
        "--json", action="store_true", help="Output as JSON")

    # GPU listing (alias for hardware)
    gpu_parser = deployments_subparsers.add_parser(
        "gpu", help="List available GPUs")
    gpu_parser.add_argument(
        "--api-key", type=str, default=None, help="API key")
    gpu_parser.add_argument(
        "--list", action="store_true", help="List available GPUs")
    gpu_parser.add_argument(
        "--json", action="store_true", help="Output as JSON")

    # For backward compatibility, if no subcommand is provided, treat as chat
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--system", default=None,
                        help="System prompt (optional)")
    parser.add_argument("--user", help="User prompt/message")
    parser.add_argument("--prompt", help="Direct prompt")
    parser.add_argument("--temperature", type=float,
                        default=None, help="Temperature")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Maximum tokens to generate")
    parser.add_argument("--stream", action="store_true", help="Stream output")
    parser.add_argument(
        "--mode", choices=["chat", "completions"], default="chat", help="API mode")

    args = parser.parse_args()

    # Handle different commands
    if args.command == "deployments":
        handle_deployments_commands(args)
    elif args.command == "chat" or (args.command is None and args.model):
        handle_chat_commands(args, parser)
    else:
        parser.print_help()


def wait_for_deployment_ready(client, deployment_id, deployment_name):
    """Wait for deployment to be ready and show status updates"""
    import time
    
    print()
    print(f"⏳ Waiting for deployment '{deployment_name}' to be ready...")
    print("   Press Ctrl+C to stop monitoring (deployment will continue in background)")
    
    try:
        while True:
            try:
                deployments = client.deployments.list()
                current_deployment = None
                
                for dep in deployments:
                    if dep.deployment_id == deployment_id:
                        current_deployment = dep
                        break
                
                if current_deployment:
                    status = current_deployment.status.lower()
                    print(f"   Status: {current_deployment.status}")
                    
                    if status in ['running', 'ready', 'active']:
                        print()
                        print("🚀 Deployment is now ready!")
                        print(f"Deployment ID: {current_deployment.deployment_id}")
                        print(f"Deployment Name: {current_deployment.deployment_name}")
                        print(f"Status: {current_deployment.status}")
                        print(f"Model: {current_deployment.model_name}")
                        print(f"Hardware: {current_deployment.hardware}")
                        break
                    elif status in ['failed', 'error', 'stopped']:
                        print()
                        print(f"❌ Deployment failed with status: {current_deployment.status}")
                        break
                    else:
                        # Still creating/pending
                        time.sleep(10)  # Wait 10 seconds before checking again
                else:
                    print("   ❌ Deployment not found")
                    break
                    
            except Exception as e:
                print(f"   Error checking status: {e}")
                time.sleep(10)
                
    except KeyboardInterrupt:
        print()
        print("⏹️  Monitoring stopped. Deployment continues in background.")
        print(f"   Check status with: gravixlayer deployments list")


def handle_deployments_commands(args):
    """Handle deployment-related commands"""
    client = GravixLayer(
        api_key=args.api_key or os.environ.get("GRAVIXLAYER_API_KEY"))

    try:
        if args.deployments_action == "create":
            print(f"Creating deployment '{args.deployment_name}' with model '{args.model_name}'...")

            # Generate unique name if auto-retry is enabled
            original_name = args.deployment_name
            if hasattr(args, 'auto_retry') and args.auto_retry:
                import random
                import string
                import time
                
                # Use timestamp + random for better uniqueness
                timestamp = str(int(time.time()))[-4:]  # Last 4 digits of timestamp
                suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
                args.deployment_name = f"{original_name}-{timestamp}{suffix}"
                print(f"Using unique name: '{args.deployment_name}'")

            try:
                response = client.deployments.create(
                    deployment_name=args.deployment_name,
                    model_name=args.model_name,
                    hardware=args.hardware,
                    min_replicas=args.min_replicas,
                    hw_type=args.hw_type
                )

                print("✅ Deployment created successfully!")
                print(f"Deployment ID: {response.deployment_id}")
                print(f"Deployment Name: {args.deployment_name}")
                print(f"Status: {response.status}")
                print(f"Model: {args.model_name}")
                print(f"Hardware: {args.hardware}")
                
                # Wait for deployment to be ready if --wait flag is used
                if hasattr(args, 'wait') and args.wait:
                    wait_for_deployment_ready(client, response.deployment_id, args.deployment_name)
                else:
                    # Add status checking
                    if hasattr(response, 'status') and response.status:
                        if response.status.lower() in ['creating', 'pending']:
                            print()
                            print("💡 Tip: Use --wait flag to monitor deployment status automatically")
                            print("   Or check status with: gravixlayer deployments list")
                        elif response.status.lower() in ['running', 'ready']:
                            print("🚀 Deployment is ready to use!")
                        
            except Exception as create_error:
                # Parse the error message to provide better feedback
                error_str = str(create_error)
                
                # Try to parse JSON error response
                try:
                    import json
                    if error_str.startswith('{"') and error_str.endswith('}'):
                        error_data = json.loads(error_str)
                        error_code = error_data.get('code', 'unknown')
                        error_message = error_data.get('error', error_str)
                        
                        # Check if deployment was actually created despite the error
                        if 'already exists' in error_message.lower():
                            try:
                                # Wait a moment for the deployment to appear
                                import time
                                time.sleep(3)
                                
                                updated_deployments = client.deployments.list()
                                deployment_found = None
                                for dep in updated_deployments:
                                    if dep.deployment_name == args.deployment_name:
                                        deployment_found = dep
                                        break
                                
                                if deployment_found:
                                    print("✅ Deployment created successfully!")
                                    print(f"Deployment ID: {deployment_found.deployment_id}")
                                    print(f"Deployment Name: {deployment_found.deployment_name}")
                                    print(f"Status: {deployment_found.status}")
                                    print(f"Model: {deployment_found.model_name}")
                                    print(f"Hardware: {deployment_found.hardware}")
                                    
                                    # Wait for deployment to be ready if --wait flag is used
                                    if hasattr(args, 'wait') and args.wait:
                                        wait_for_deployment_ready(client, deployment_found.deployment_id, deployment_found.deployment_name)
                                    return
                                else:
                                    print(f"❌ Deployment creation failed: {error_message}")
                                    if hasattr(args, 'auto_retry') and args.auto_retry:
                                        print("Auto-retry was already attempted but failed.")
                                    else:
                                        print(f"Try with --auto-retry flag: gravixlayer deployments create --deployment_name \"{original_name}\" --hardware \"{args.hardware}\" --model_name \"{args.model_name}\" --auto-retry")
                            except Exception:
                                print(f"❌ Deployment creation failed: {error_message}")
                        else:
                            print(f"❌ Deployment creation failed: {error_message}")
                    else:
                        print(f"❌ Deployment creation failed: {error_str}")
                except (json.JSONDecodeError, ValueError):
                    print(f"❌ Deployment creation failed: {error_str}")
                return

        elif args.deployments_action == "list":
            deployments = client.deployments.list()

            if args.json:
                print(json.dumps([d.model_dump()
                      for d in deployments], indent=2))
            else:
                if not deployments:
                    print("No deployments found.")
                else:
                    print(f"Found {len(deployments)} deployment(s):")
                    print()
                    for deployment in deployments:
                        print(f"Deployment ID: {deployment.deployment_id}")
                        print(f"Deployment Name: {deployment.deployment_name}")
                        print(f"Model: {deployment.model_name}")
                        print(f"Status: {deployment.status}")
                        print(f"Hardware: {deployment.hardware}")
                        print(f"Replicas: {deployment.min_replicas}")
                        print(f"Created: {deployment.created_at}")
                        print()

        elif args.deployments_action == "delete":
            print(f"Deleting deployment {args.deployment_id}...")
            response = client.deployments.delete(args.deployment_id)
            print("Deployment deleted successfully!")
            print(f"   Response: {response}")

        elif args.deployments_action in ["hardware", "gpu"]:
            if hasattr(args, 'list') and args.list:
                accelerators = client.accelerators.list()
                
                if hasattr(args, 'json') and getattr(args, 'json', False):
                    import json as json_module
                    # Filter out unwanted fields from JSON output
                    filtered_accelerators = []
                    for a in accelerators:
                        data = a.model_dump()
                        # Remove the specified fields
                        data.pop('name', None)
                        data.pop('memory', None)
                        data.pop('gpu_type', None)
                        data.pop('use_case', None)
                        filtered_accelerators.append(data)
                    print(json_module.dumps(filtered_accelerators, indent=2))
                else:
                    if not accelerators:
                        print("No accelerators/GPUs found.")
                    else:
                        print(f"Available {'Hardware' if args.deployments_action == 'hardware' else 'GPUs'} ({len(accelerators)} found):")
                        print()
                        print(f"{'Accelerator':<15} {'Hardware String':<35} {'Memory':<10}")
                        print("-" * 60)
                        
                        for accelerator in accelerators:
                            gpu_type = getattr(accelerator, 'gpu_type', accelerator.name)
                            hardware_string = accelerator.hardware_string
                            memory = getattr(accelerator, 'memory', 'N/A')
                            
                            print(f"{gpu_type:<15} {hardware_string:<35} {memory:<10}")
            else:
                print(f"Use --list flag to list available {'hardware' if args.deployments_action == 'hardware' else 'GPUs'}")
                print(f"Example: gravixlayer deployments {args.deployments_action} --list")

    except Exception as e:
        print(f"Error: {e}")


def handle_chat_commands(args, parser):
    """Handle chat and completion commands"""
    # Validate arguments
    if args.mode == "chat" and not args.user:
        parser.error("--user is required for chat mode")
    if args.mode == "completions" and not args.prompt:
        parser.error("--prompt is required for completions mode")

    client = GravixLayer(
        api_key=args.api_key or os.environ.get("GRAVIXLAYER_API_KEY"))

    try:
        if args.mode == "chat":
            # Chat completions mode
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            messages.append({"role": "user", "content": args.user})

            if args.stream:
                for chunk in client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    stream=True
                ):
                    if chunk.choices[0].delta.content:
                        print(chunk.choices[0].delta.content,
                              end="", flush=True)
                print()
            else:
                completion = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens
                )
                print(completion.choices[0].message.content)

        else:
            # Text completions mode
            if args.stream:
                for chunk in client.completions.create(
                    model=args.model,
                    prompt=args.prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    stream=True
                ):
                    if chunk.choices[0].text:
                        print(chunk.choices[0].text, end="", flush=True)
                print()
            else:
                completion = client.completions.create(
                    model=args.model,
                    prompt=args.prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens
                )
                print(completion.choices[0].text)

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
