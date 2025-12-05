from django.contrib import admin
from .models import ImageGeneration


@admin.register(ImageGeneration)
class ImageGenerationAdmin(admin.ModelAdmin):
    list_display = ['id', 'user_text', 'target_style', 'created_at']
    list_filter = ['created_at']
    search_fields = ['user_text', 'target_style', 'processed_prompt']
    readonly_fields = ['created_at']
    fields = ['user_text', 'target_style', 'processed_prompt', 'ad_json', 'image_path', 'created_at']
    
    def get_readonly_fields(self, request, obj=None):
        return self.readonly_fields + ['ad_json'] if obj else self.readonly_fields

