from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import json
import threading
from .services.llm_service import LLMService
from .services.diffusion_service import DiffusionService
from .models import ImageGeneration


# Get model name from settings
model_name = getattr(settings, 'GPT4ALL_MODEL_NAME', 'DeepSeek-R1-Distill-Qwen-7B')
llm_service = LLMService(model_name=model_name)
diffusion_service = DiffusionService()


def index(request):
    """Main page"""
    return render(request, 'generator/index.html')


@csrf_exempt
@require_http_methods(["POST"])
def generate_image(request):
    """
    API endpoint for generating images
    
    Receives JSON data:
    {
        "user_text": "User input text"
    }
    
    Returns JSON data:
    {
        "success": true,
        "image_url": "Generated image URL",
        "ad_json": "Generated ad JSON object"
    }
    """
    try:
        # Parse request data
        data = json.loads(request.body)
        user_text = data.get('user_text', '').strip()
        
        if not user_text:
            return JsonResponse({
                'success': False,
                'error': 'User input text cannot be empty'
            }, status=400)
        
        # Use LLM to generate ad JSON (target_style is automatically determined by LLM)
        ad_info = llm_service.process_prompt(user_text, None)
        ad_json_str = json.dumps(ad_info, ensure_ascii=False, indent=2)
        
        # Extract image generation prompt and style from JSON
        processed_prompt = ad_info.get('ad_description', f"{user_text}, high quality, detailed")
        target_style = ad_info.get('style', 'default')
        ad_copy = ad_info.get('ad_copy', '')
        text_layout = ad_info.get('text_layout', None)
        
        # Create database record first (without image_path)
        generation_record = ImageGeneration.objects.create(
            user_text=user_text,
            target_style=target_style,
            processed_prompt=processed_prompt,
            ad_json=ad_json_str,
            image_path=''  # Will be updated when image is generated
        )
        
        # Return JSON immediately, generate image in background
        response_data = {
            'success': True,
            'ad_json': ad_info,
            'record_id': generation_record.id,
            'image_status': 'generating',  # Indicate image is being generated
            'image_url': None
        }
        
        # Generate image in background thread
        def generate_image_async():
            try:
                image_url = diffusion_service.generate_image_from_prompt(
                    style=target_style,
                    ad_copy=ad_copy,
                    text_layout=text_layout
                )
                # Update database record with image path
                generation_record.image_path = image_url
                generation_record.save(update_fields=['image_path'])
                print(f"✓ Image generated and saved: {image_url}")
            except Exception as e:
                print(f"✗ Error generating image in background: {e}")
                # Update record with error status (optional)
                generation_record.image_path = f"ERROR: {str(e)}"
                generation_record.save(update_fields=['image_path'])
        
        # Start image generation in background
        image_thread = threading.Thread(target=generate_image_async, daemon=True)
        image_thread.start()
        
        return JsonResponse(response_data)
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error generating image: {str(e)}'
        }, status=500)


@require_http_methods(["GET"])
def get_record_status(request, record_id):
    """Get status of a specific generation record"""
    try:
        record = ImageGeneration.objects.get(id=record_id)
        ad_json_dict = record.get_ad_json_dict()
        
        return JsonResponse({
            'success': True,
            'id': record.id,
            'user_text': record.user_text,
            'target_style': record.target_style,
            'processed_prompt': record.processed_prompt,
            'ad_json': ad_json_dict,
            'image_url': record.image_path if record.image_path and not record.image_path.startswith('ERROR:') else None,
            'image_status': 'ready' if record.image_path and not record.image_path.startswith('ERROR:') else 'generating' if not record.image_path else 'error',
            'created_at': record.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    except ImageGeneration.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Record not found'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def history(request):
    """Get generation history records"""
    records = ImageGeneration.objects.all()[:20]  # Last 20 records
    
    history_data = []
    for record in records:
        ad_json_dict = record.get_ad_json_dict()
        history_data.append({
            'id': record.id,
            'user_text': record.user_text,
            'target_style': record.target_style,
            'processed_prompt': record.processed_prompt,
            'ad_json': ad_json_dict,
            'image_url': record.image_path if record.image_path and not record.image_path.startswith('ERROR:') else None,
            'image_status': 'ready' if record.image_path and not record.image_path.startswith('ERROR:') else 'generating' if not record.image_path else 'error',
            'created_at': record.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return JsonResponse({
        'success': True,
        'history': history_data
    })

