"""
Task handlers for the download workflow: index building, downloading, and
(placeholder) validation.
"""

import asyncio
import logging

from src.data.common.hpc.client import HPCClient

logger = logging.getLogger(__name__)


class TaskHandlers:
    """Unified task handlers for all workflow operations."""

    @staticmethod
    def handle_index(data_source, download_index, context, task_config):
        """Handle index-building task."""
        logger.info(f"Building index for {data_source.DATA_SOURCE_NAME}")

        # Extract parameters from task config
        rebuild = task_config.get('rebuild', False)
        only_missing_entrypoints = task_config.get('only_missing_entrypoints', True)
        sync_direction = task_config.get('sync_direction', 'auto')

        try:
            # Check for schema conversion setting
            force_schema_conversion = task_config.get('force_schema_conversion', False)

            # If schema conversion is forced, rebuild the index
            if force_schema_conversion:
                logger.info("Force schema conversion enabled - rebuilding index")
                rebuild = True

            # Sync index first if configured and HPC target is available
            if hasattr(context, 'hpc_target') and context.hpc_target and sync_direction != 'none':
                success = download_index.ensure_synced_index(
                    hpc_target=context.hpc_target,
                    sync_direction=sync_direction,
                    key_file=context.key_file
                )
                if not success:
                    logger.warning("Index sync failed, continuing with local index")

            # Get build_index_from_source parameters
            build_params = {
                'data_source': data_source,
                'rebuild': rebuild,
                'only_missing_entrypoints': only_missing_entrypoints
            }

            # Add schema parameters only if the method accepts them
            import inspect
            build_index_sig = inspect.signature(download_index.build_index_from_source)
            if 'schema_dtypes' in build_index_sig.parameters:
                build_params['schema_dtypes'] = getattr(data_source, 'schema_dtypes', {})
            if 'force_schema_conversion' in build_index_sig.parameters:
                build_params['force_schema_conversion'] = force_schema_conversion

            # Build index from source with appropriate parameters
            try:
                files_indexed = download_index.build_index_from_source(**build_params)
            except ValueError as e:
                if "Schema" in str(e) or "migrate" in str(e):
                    logger.warning(f"Schema migration error: {e}")
                    logger.info("Attempting to rebuild index with schema conversion")
                    # Force rebuild on schema errors
                    build_params['rebuild'] = True
                    files_indexed = download_index.build_index_from_source(**build_params)
                else:
                    raise

            logger.info(f"Index building complete: {files_indexed} files indexed")

            # Save index
            download_index.save()

            # Sync back to HPC if configured
            if (hasattr(context, 'hpc_target') and context.hpc_target and
                sync_direction in ['auto', 'push']):
                download_index.sync_index_with_hpc(
                    hpc_target=context.hpc_target,
                    direction='push',
                    key_file=context.key_file
                )

            return True

        except Exception as e:
            logger.error(f"Error in index task: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return False

    @staticmethod
    def handle_download(data_source, download_index, context, task_config):
        """Handle download task using async downloader."""
        logger.info("Starting async download task")

        try:
            # Import the async downloader
            from src.data.download.async_downloader import run_async_download_workflow

            # Check if we have HPC context
            if hasattr(context, 'hpc_target') and context.hpc_target:
                # Ensure index is synced before downloads
                logger.info("Ensuring index is synced before downloads")
                sync_success = download_index.ensure_synced_index(
                    hpc_target=context.hpc_target,
                    sync_direction='pull',  # Always pull before downloads
                    key_file=context.key_file
                )

                if not sync_success:
                    logger.warning("Index sync failed, continuing with local index")

                # Create HPC client
                hpc_client = HPCClient(
                    target=context.hpc_target,
                    key_file=context.key_file
                )

                # Run async download workflow
                return asyncio.run(run_async_download_workflow(
                    data_source=data_source,
                    index=download_index,
                    hpc_client=hpc_client,
                    context=context,
                    config=task_config
                ))
            else:
                logger.warning("Download requires HPC target configuration")
                return False

        except ImportError as e:
            logger.error(f"Error importing async downloader: {e}")
            return False
        except Exception as e:
            logger.error(f"Error in download task: {e}")
            return False

    @staticmethod
    def handle_validate(data_source, download_index, context, task_config):
        """Handle validation task - PLACEHOLDER."""
        logger.warning("Validation workflow functionality has been removed")
        logger.info("This is a placeholder for future implementation")

        # TODO: Implement validation workflow
        # - Verify downloaded files
        # - Check batch integrity
        # - Validate transfers to HPC
        # - Report status

        return False  # Not implemented
