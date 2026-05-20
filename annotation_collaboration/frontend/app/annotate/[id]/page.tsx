import { Annotator } from "@/components/Annotator";

export default async function AnnotatePage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <Annotator videoId={id} />;
}
